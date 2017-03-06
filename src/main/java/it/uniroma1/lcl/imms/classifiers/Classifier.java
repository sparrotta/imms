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
import java.text.ParseException;
import java.util.Collection;
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
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.FeaturesAnnotation;
import it.uniroma1.lcl.imms.Constants.HeadsAnnotation;
import it.uniroma1.lcl.imms.Constants.LexicalItemAnnotation;
import it.uniroma1.lcl.imms.annotator.feature.Feature;

public abstract class Classifier<M>  {

	String fileSeparator = System.getProperty("file.separator");
	
	public static final String CUSTOM_FEATURIZER_PREFIX = "customFeaturizerClass.";
	private Properties properties;
	
	private Map<String, RVFDataset<String,String>> datasets = new HashMap<String,RVFDataset<String,String>>();
	private Map<String,Map<String,Index>> lexElementFeatureValueIndices = new HashMap<String,Map<String,Index>>();
	
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
		RVFDataset<String,String> d = datasets.get(lexElement);
		if(d==null){
			if(getModelDir()!=null && getStatDir()!=null){
				try {
					readModel(lexElement, getModelDir());
					readStat(lexElement,getStatDir());
					d = datasets.get(lexElement);
					if(d!=null){
						d.featureIndex().lock();
					}
				} catch (ClassNotFoundException | IOException | ParseException e) {
					throw new IllegalArgumentException("No suitable model or stat for lexical element: "+lexElement);
				}
			} else {
				d = new RVFDataset<String,String>();
				datasets.put(lexElement,d);
			}						
		}
		
		if(d==null){
			throw new RuntimeException("No dataset available for lexical element: "+lexElement);
			
		}
		
		Counter<String> counter = new ClassicCounter<String>();
		for(Feature feature : head.get(FeaturesAnnotation.class)){
			counter.setCount(feature.key(),toDouble(feature.value()));
		}
		String label = head.get(WordSenseAnnotation.class);
		String src = head.get(Constants.DocSourceAnnotation.class);
		String id = head.get(IDAnnotation.class);
		
		d.add(new RVFDatum<String,String>(counter, label),src,id);
	}

	protected Counter<String> asFeatureCounter(CoreMap head){
		ClassicCounter<String> counter = new ClassicCounter<String>();
		for(Feature feature : head.get(FeaturesAnnotation.class)){
			counter.setCount(feature.key(), getFeatureValue(head.get(LexicalItemAnnotation.class),feature));
		}
		return counter;
	}
	
	double getFeatureValue(String lexElem,Feature feature){		
		Object featureValue = feature.value();		
		if(featureValue instanceof Number){
			return ((Number)featureValue).doubleValue();
		}				
		Map<String, Index> fvi = lexElementFeatureValueIndices.get(lexElem);
		if(fvi==null){
			fvi=new HashMap<String,Index>();
			lexElementFeatureValueIndices.put(lexElem,fvi);
		}
		Index featureValuesIndex = fvi.get(feature.key());
		if(!dataset(lexElem).featureIndex().isLocked() && featureValuesIndex==null){
			featureValuesIndex=new HashIndex<>();
			fvi.put(feature.key(), featureValuesIndex);			
		}							
		return featureValuesIndex == null ? -1.0 : featureValuesIndex.addToIndex(featureValue);
	}
	
	
	public Properties getProperties() {
		return properties;
	}

	
	public RVFDataset<String,String> dataset(String lexElement){		
		return datasets.get(lexElement);
	}
	
	public Map<String,RVFDataset<String,String>> allDatasets(){		
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
		RVFDataset<String,String> d = datasets.get(lexElement);
		if(d==null){
			d = new RVFDataset<String,String>();
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
			offset = line.length();
			line = br.readLine(); 
			if(line!=null){
				for(String feat : line.split(sep)){
					d.featureIndex.add(feat);
				}
			} else {
				throw new ParseException("Stat file bad format",offset);
			}
		} finally{
			br.close();
		}
		d.featureIndex().lock();
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
		RVFDataset<String,String> d = datasets.get(lexElement);		
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
			if(i>0){os.append(sep);}
			os.append(featureName);
		}	
		os.append("\n");
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