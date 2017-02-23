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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.HeadAnnotation;
import it.uniroma1.lcl.imms.Constants.LexicalElementAnnotation;
import it.uniroma1.lcl.imms.feature.Feature;

public abstract class Classifier<M>  {

	String fileSeparator = System.getProperty("file.separator");
	
	public static final String CUSTOM_FEATURIZER_PREFIX = "customFeaturizerClass.";
	private Properties properties;
	

	List<String> ids = new ArrayList<String>();
	private Map<String, Dataset<String,Feature<?>>> datasets = new HashMap<String,Dataset<String,Feature<?>>>();
	private Map<String, List<String>> idsMap = new HashMap<String,List<String>>();
	private Map<String,M> models = new HashMap<String,M>();
	
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
	
	public void add(Annotation anno, String modelDir, String statDir) {
		String lexElement = anno.get(LexicalElementAnnotation.class);
		Dataset d = datasets.get(lexElement);
		if(d==null){			
			try {
				readModel(lexElement, modelDir);
				readStat(lexElement,statDir);
			} catch (ClassNotFoundException | IOException | ParseException e) {
				throw new IllegalArgumentException("No suitable model or stat for lexical element: "+lexElement);
			}
		}		
		List<String> ids = idsMap.get(lexElement);
		if(ids==null){
			ids = new ArrayList<String>();
			idsMap.put(lexElement,ids);
		}
		ids.add(anno.get(DocIDAnnotation.class));
		CoreLabel head = anno.get(HeadAnnotation.class);		
		d.add(head.get(Constants.FeaturesAnnotation.class), head.getString(WordSenseAnnotation.class));	
	}
	
	public void add(Annotation anno) {
		String lexElement = anno.get(LexicalElementAnnotation.class);
		Dataset d = datasets.get(lexElement);
		if(d==null){
			d = new Dataset<>();
			datasets.put(lexElement,d);
		}		
		List<String> ids = idsMap.get(lexElement);
		if(ids==null){
			ids = new ArrayList<String>();
			idsMap.put(lexElement,ids);
		}
		ids.add(anno.get(DocIDAnnotation.class));
		CoreLabel head = anno.get(HeadAnnotation.class);		
		d.add(head.get(Constants.FeaturesAnnotation.class), head.getString(WordSenseAnnotation.class));	
	}
	
	public Properties getProperties() {
		return properties;
	}

	public Dataset<String,Feature<?>> dataset(String lexElement){		
		return datasets.get(lexElement);
	}

	public M model(String lexElement){				
		return models.get(lexElement);
	}
	
	
	public List<String> ids(String lexElement){
		return idsMap.get(lexElement);
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
		Dataset d = datasets.get(lexElement);
		if(d==null){
			d = new Dataset();
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
					d.featureIndex.add(new Feature(feat,true));
				}
			} else {
				throw new ParseException("Stat file bad format",offset);
			}
		} finally{
			br.close();
		}
	}

	public void write(String outDir) throws IOException{
		for(String lexElem : datasets.keySet()){
			writeStat(lexElem,outDir);
		}
		for(String lexElem : models.keySet()){
			writeModel(lexElem,outDir);
		}
	}
	void writeStat(String lexElement, String outDir) throws IOException {
		Dataset<String,Feature<?>> d = datasets.get(lexElement);
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
			if (i > 0) {
				os.append(sep);
			}
			os.append(d.featureIndex.get(i).key());
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
}