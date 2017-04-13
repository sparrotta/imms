package it.uniroma1.lcl.imms.classifiers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.CoreAnnotations.IDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.FeaturesAnnotation;
import it.uniroma1.lcl.imms.Constants.HeadsAnnotation;
import it.uniroma1.lcl.imms.Constants.LexicalItemAnnotation;
import it.uniroma1.lcl.imms.annotator.feature.Feature;
import net.didion.jwnl.JWNL;
import net.didion.jwnl.JWNLException;
import net.didion.jwnl.data.IndexWord;
import net.didion.jwnl.dictionary.Dictionary;

public abstract class Classifier<M> {

	String fileSeparator = System.getProperty("file.separator");

	public static final String CUSTOM_FEATURIZER_PREFIX = "customFeaturizerClass.";
	private Properties properties;

	private Map<String, M> models = new HashMap<String, M>();

	private ClassifierData trainingData = new ClassifierData();
	private ClassifierData testData = new ClassifierData();

	private String modelDir;

	private String statDir;

	public Classifier(Properties properties) {
		this.properties = properties;
	}

	public Map<String, RVFDataset<String, String>> getTestData(){
		return testData.allDatasets();
	}
	
	public RVFDataset<String, String> getTestData(String lexElement) {
		return testData.dataset(lexElement);
	}

	public RVFDataset<String, String> getTrainingData(String lexElement) {
		return trainingData.dataset(lexElement);
	}	

	
	public void addTrainingSample(Annotation anno) {
		for (CoreMap head : anno.get(HeadsAnnotation.class)) {
			String lexElement = head.get(LexicalItemAnnotation.class);
			RVFDataset<String, String> d = trainingData.datasets.get(lexElement);
			if (d == null) {
				d = new RVFDataset<String, String>();				
				trainingData.datasets.put(lexElement, d);
			}
			trainingData.addDatum(head, d);
		}
	}

	
	public void addTestSample(Annotation anno) {
		for (CoreMap head : anno.get(HeadsAnnotation.class)) {
			String lexElement = head.get(LexicalItemAnnotation.class);
			RVFDataset<String, String> d = testData.datasets.get(lexElement);
			if (d == null) {
				d = readStat(lexElement, getStatDir());
				testData.datasets.put(lexElement, d);
			}
			testData.addDatum(head, d);
		}

	}

	public Properties getProperties() {
		return properties;
	}

	public M getModel(String lexElement) {
		M model = models.get(lexElement);
		if (model == null) {
			try {
				model = readModel(lexElement, getModelDir());
				models.put(lexElement, model);
			} catch (ClassNotFoundException | IOException e) {
			}
		}
		return model;
	}

	public void train() {
		for (String lexElem : trainingData.lexicalElements()) {
			this.models.put(lexElem, train(lexElem));
		}
	}

	public Map<String, List<String>> test() {
		Map<String, List<String>> answersMap = new HashMap<String, List<String>>();
		for (String lexElem : testData.lexicalElements()) {
			RVFDataset<String,String> d = getTestData(lexElem);
			M model = getModel(lexElem);			
			if(model!=null){				
				answersMap.put(lexElem, test(d,model));
			} else {
				List<String> answers = new ArrayList<String>();					
				for (int i = 0; i < d.size(); i++) {					
					answers.add(null);											
				}
				answersMap.put(lexElem, answers);
			}
		}
		return answersMap;
	}

	public abstract M train(String lexElem);

	public abstract List<String> test(RVFDataset<String,String> d, M model);

	M readModel(String lexElement, String dir) throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream ois = new ObjectInputStream(
				new GZIPInputStream(new FileInputStream(dir + fileSeparator + lexElement + ".model.gz")));
		return (M) ois.readObject();
	}

	RVFDataset<String, String> readStat(String lexElement, String dir) {

		RVFDataset<String, String> d = null;

		BufferedReader br = null;
		try {

			d = new RVFDataset<String, String>();
			br = new BufferedReader(new InputStreamReader(
					new GZIPInputStream(new FileInputStream(dir + fileSeparator + lexElement + ".stat.gz"))));

			String sep = "\t";
			int offset = 0;
			String line = br.readLine();

			if (line != null) {
				for (String label : line.split(sep)) {
					d.labelIndex.add(label);
				}
			} else {
				throw new IOException("Stat file empty");
			}
			offset = line.length();
			line = br.readLine();
			if (line != null) {
				for (String feat : line.split(sep)) {
					d.featureIndex.add(feat);
				}
			} else {
				throw new IOException("Stat file bad format");
			}
		} catch (IOException e) {
			d = new RVFDataset<String, String>();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {

				}
			}
		}
		d.featureIndex.lock();
		return d;
	}

	public void write(String modelDir, String statDir) throws IOException {
		for (String lexElem : models.keySet()) {
			writeModel(lexElem, modelDir);
		}
		for (String lexElem : trainingData.lexicalElements()) {
			writeStat(lexElem, statDir);
		}
	}

	void writeStat(String lexElement, String outDir) throws IOException {
		RVFDataset<String, String> d = trainingData.datasets.get(lexElement);
		if (d == null) {
			return;
		}
		OutputStreamWriter os = new OutputStreamWriter(
				new GZIPOutputStream(new FileOutputStream(outDir + fileSeparator + lexElement + ".stat.gz")));
		String sep = "\t";
		for (int i = 0; i < d.labelIndex.size(); i++) {
			if (i > 0) {
				os.append(sep);
			}
			os.append(d.labelIndex.get(i));
		}
		os.append("\n");
		for (int i = 0; i < d.featureIndex.size(); i++) {
			String featureName = d.featureIndex.get(i);
			if (i > 0) {
				os.append(sep);
			}
			os.append(featureName);
		}
		os.append("\n");
		os.flush();
		os.close();
	}

	void writeModel(String lexElement, String outDir) throws FileNotFoundException, IOException {
		M model = this.models.get(lexElement);
		if (model != null) {
			ObjectOutputStream oos = new ObjectOutputStream(
					new GZIPOutputStream(new FileOutputStream(outDir + fileSeparator + lexElement + ".model.gz")));
			oos.writeObject(model);
			oos.flush();
			oos.close();
		}
	}

	public String getModelDir() {
		return modelDir;
	}

	public void setModelDir(String modelDir) {
		this.modelDir = modelDir;
	}

	public String getStatDir() {
		return statDir;
	}

	public void setStatDir(String statDir) {
		this.statDir = statDir;
	}

	private class ClassifierData {

		private Map<String, RVFDataset<String, String>> datasets = new HashMap<String, RVFDataset<String, String>>();

		public Set<String> lexicalElements() {
			return datasets.keySet();
		}

		protected Double toDouble(Object o) {
			Double d = 0.0;
			if (o instanceof Boolean) {
				d = ((Boolean) o) ? 1.0 : 0.0;
			} else if (o instanceof Number) {
				d = ((Number) o).doubleValue();
			} else if (o instanceof String) {
				d = Double.valueOf((String) o);
			}
			return d;
		}

		private void addDatum(CoreMap head, RVFDataset<String, String> d) {
			Counter<String> counter = new ClassicCounter<String>();
			for (Feature feature : head.get(FeaturesAnnotation.class)) {
				counter.setCount(feature.key(), toDouble(feature.value()));
			}
			String label = head.get(WordSenseAnnotation.class);
			String src = head.get(Constants.DocSourceAnnotation.class);
			String id = head.get(IDAnnotation.class);

			d.add(new RVFDatum<String, String>(counter, label), src, id);
		}

		public RVFDataset<String, String> dataset(String lexElement) {
			return datasets.get(lexElement);
		}

		public Map<String, RVFDataset<String, String>> allDatasets() {
			return datasets;
		}

		public String src(String lexElement, int i) {
			return datasets.get(lexElement).getRVFDatumSource(i);
		}

		public String id(String lexElement, int i) {
			return datasets.get(lexElement).getRVFDatumId(i);
		}

	}
}