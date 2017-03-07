package it.uniroma1.lcl.imms.classifiers;

import edu.stanford.nlp.ling.CoreAnnotations.IDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.FeaturesAnnotation;
import it.uniroma1.lcl.imms.Constants.HeadsAnnotation;
import it.uniroma1.lcl.imms.Constants.LexicalItemAnnotation;
import it.uniroma1.lcl.imms.annotator.feature.Feature;

import java.io.*;
import java.text.ParseException;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public abstract class Classifier<M>  {

	String fileSeparator = System.getProperty("file.separator");
	
	public static final String CUSTOM_FEATURIZER_PREFIX = "customFeaturizerClass.";
	private Properties properties;
	
	private Map<String, IMMSDataset> datasets = new HashMap<String,IMMSDataset>();
	
	private Map<String,M> models = new HashMap<String,M>();

	private String modelDir;

	private String statDir;
	
	public Classifier(Properties properties) {
		this.properties = properties;
	}

	public Set<String> lexicalElements(){
		return datasets.keySet();
	}
	
	public void loadFromFiles(String lexElement, String modelDir, String statDir) throws FileNotFoundException, IOException, ParseException, ClassNotFoundException {	
		readModel(lexElement, modelDir);
		readStat(lexElement, statDir);
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
	
	public void add(Annotation anno) {						
		for(CoreMap head : anno.get(HeadsAnnotation.class)){
			addDatum(head);
		}				
	}
	
	private void addDatum(CoreMap head){
		String lexElement = head.get(LexicalItemAnnotation.class);
		IMMSDataset d = datasets.get(lexElement);
		if(d==null){
			if(getModelDir()!=null && getStatDir()!=null){
				try {
					readModel(lexElement, getModelDir());
					readStat(lexElement,getStatDir());
					d = datasets.get(lexElement);
					if(d!=null){
						d.featureIndex.lock();
					}
				} catch (ClassNotFoundException | IOException | ParseException e) {
					throw new IllegalArgumentException("No suitable model or stat for lexical element: "+lexElement);
				}
			} else {
				d = new IMMSDataset();
				datasets.put(lexElement,d);
			}						
		}
		
		if(d==null){
			throw new RuntimeException("No dataset available for lexical element: "+lexElement);
			
		}
		
		Collection<Feature>features =  head.get(FeaturesAnnotation.class);
		String label = head.get(WordSenseAnnotation.class);
		String src = head.get(Constants.DocSourceAnnotation.class);
		String id = head.get(IDAnnotation.class);
		
		d.add(features,label,src,id);
	}

	
	
	
	
	public Properties getProperties() {
		return properties;
	}

	
	public IMMSDataset dataset(String lexElement){		
		return datasets.get(lexElement);
	}
	
	public Map<String,IMMSDataset> allDatasets(){		
		return datasets;
	}

	public M model(String lexElement){				
		return models.get(lexElement);
	}
	
	
	public String src(String lexElement, int i){
		return datasets.get(lexElement).getRVFDatumSource(i);
	}
	
	public String id(String lexElement, int i){
		return datasets.get(lexElement).getRVFDatumId(i);
	}
	
	public void train(){
		for(String lexElem :datasets.keySet()){
			this.models.put(lexElem, train(lexElem));
		}
	}
	public Map<String, List<String>> test(){		
		Map<String,List<String>>answersMap =new HashMap<String,List<String>>(); 
		for(String lexElem :datasets.keySet()){			
			answersMap.put(lexElem, test(lexElem));
		}		
		return answersMap;
	}
	


	public abstract M train(String lexElem);
	
	public abstract List<String> test(String lexElem);

	void readModel(String lexElement,String dir) throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(new FileInputStream(dir+fileSeparator+lexElement+".model.gz")));		
		this.models.put(lexElement, (M) ois.readObject());		
	}
	void readStat(String lexElement,String dir) throws FileNotFoundException, IOException, ParseException {
		IMMSDataset d = datasets.get(lexElement);
		if(d==null){
			d = new IMMSDataset();
			datasets.put(lexElement, d);
		}
		
		BufferedReader br = new BufferedReader(
				new InputStreamReader(new GZIPInputStream(new FileInputStream(dir+fileSeparator+lexElement+".stat.gz"))));
		String sep = "\t";
		int offset = 0;
		String line = br.readLine();
		try {
			if(line!=null){
				for(String label : line.split(sep)){
					d.labelIndex.add(label);					
				}
			} else {
				throw new ParseException("Stat file empty",0);
			}
			while((line=br.readLine())!=null){
				offset += line.length();
				String[] featureValuesArr=line.split(sep);
				int fID = d.featureIndex.addToIndex(featureValuesArr[0]);
				Counter<String> featureValues = new ClassicCounter<String>();
				for(int i=1; i<featureValuesArr.length;i++){					
					featureValues.incrementCount(featureValuesArr[i]);
				}
				d.featureValuesMap.put(fID, featureValues);					
			}
			if(d.featureIndex.size()==0) {
				throw new ParseException("Stat file bad format. Missing features",offset);
			}
		} finally{
			br.close();
		}
		d.featureIndex.lock();
	}

	public void write(String modelDir,String statDir) throws IOException{		
		for(String lexElem : models.keySet()){
			writeModel(lexElem,modelDir);
		}
		for(String lexElem : datasets.keySet()){
			writeStat(lexElem,statDir);
		}
	}
	void writeStat(String lexElement, String outDir) throws IOException {
		IMMSDataset d = datasets.get(lexElement);		
		if(d==null){
			return;
		}
		OutputStreamWriter os = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(outDir+fileSeparator+lexElement+".stat.gz")));
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
			os.append(featureName);
			Counter<String> featureValues = d.featureValuesMap.get(i);
			if(featureValues!=null){
				for(String key : featureValues.keySet()){
					os.append("\t");
					os.append(key);
				}
			}			
			os.append("\n");
		}	
		
		os.flush();
		os.close();		
	}
	
	void writeModel(String lexElement,String outDir) throws FileNotFoundException, IOException{
		M model = this.models.get(lexElement);
		if (model != null) {
			ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(outDir+fileSeparator+lexElement+".model.gz")));
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
}