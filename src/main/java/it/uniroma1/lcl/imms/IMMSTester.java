package it.uniroma1.lcl.imms;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;

import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.pipeline.Annotation;
import it.uniroma1.lcl.imms.annotator.IMMSPipeline;
import it.uniroma1.lcl.imms.classifiers.Classifier;
import it.uniroma1.lcl.imms.task.ITaskHandler;

public class IMMSTester {

	
	private IMMSPipeline pipeline;
	private Properties properties;

	public IMMSTester(Properties props) {
		this.properties = props;
		this.pipeline = new IMMSPipeline(props);		
	}

	

	void doTest(String testFile) throws FileNotFoundException, IOException {
		ITaskHandler th = this.pipeline.getTaskHandler();
		th.loadCorpus(testFile);
		doTest(th);				
	}
	void doTest(String testFile, String keyFile) throws FileNotFoundException, IOException {
		ITaskHandler th = this.pipeline.getTaskHandler();
		th.loadCorpus(testFile);
		th.loadAnswers(keyFile);
		doTest(th);				
	}
	void doTest(ITaskHandler taskHandler) {				
		Iterator<Annotation> it = taskHandler.iterator();
		Classifier classifier = pipeline.getClassifier();
		classifier.setModelDir(properties.getProperty(Constants.PROPERTY_CLASSIFIER_MODEL_DIR));
		classifier.setStatDir(properties.getProperty(Constants.PROPERTY_CLASSIFIER_STAT_DIR));		
		
		while (it.hasNext()) {
			Annotation text = it.next();
			pipeline.annotate(text);
			classifier.add(text);
		}
		
		classifier.test();		
		for(Entry<String, RVFDataset<String, String>>entry : ((Map<String, RVFDataset<String,String>>)classifier.allDatasets()).entrySet()){						
			taskHandler.writeResults(properties.getProperty(Constants.PROPERTY_TASK_RESULT_DIR),entry.getKey(),entry.getValue());
		}
		
	}
	
	public static void main(String[] args) {	
		
		String testFilename = args[0];
		String keyFilename = testFilename+".key";
		
		//Just as a reminder of original ims command line
		//TODO porting from command line into properties
		String generalOptions = "Usage: testPath modelDir statisticDir saveDir\n"
				+ "\t-i class name of Instance Extractor(default sg.edu.nus.comp.nlp.ims.instance.CInstanceExtractor)\n"
				+ "\t-f class name of Feature Extractor(default sg.edu.nus.comp.nlp.ims.feature.CFeatureExtractorCombination)\n"
				+ "\t-c class name of Corpus(default sg.edu.nus.comp.nlp.ims.corpus.CLexicalCorpus)\n"
				+ "\t-e class name of Evaluator(default sg.edu.nus.comp.nlp.ims.classifiers.CLibLinearEvaluator)\n"
				+ "\t-r class name of Result Writer(default sg.edu.nus.comp.nlp.ims.io.CResultWriter)\n"
				+ "\t-lexelt path of lexelt file\n"
				+ "\t-is path of index.sense(option)\n"
				+ "\t-prop path of prop.xml for JWNL\n"
				+ "\t-split 1/0 whether the corpus is sentence splitted(default 0)\n"
				+ "\t-ssm path of sentence splitter model\n"
				+ "\t-token 1/0 whether the corpus is tokenized(default 0)\n"
				+ "\t-pos 1/0 whether the pos tag is provided in corpus(default 0)\n"
				+ "\t-ptm path POS tagger model\n"
				+ "\t-dict path of dictionary for opennlp POS tagger(option)\n"
				+ "\t-tagdict path of tagdict for POS tagger(option)\n"
				+ "\t-lemma 1/0 whether the lemma is provided in the corpus(default 0)\n"
				+ "\t-delimiter the delimiter to separate tokens, lemmas and POS tags (default \"/\")\n"
				+ "\t-type type of testPath\n"
				+ "\t\tdirectory: test all xml files under directory testPath\n"
				+ "\t\tlist: test all files listed in file testPath\n"
				+ "\t\tfile(default): test file testPath\n";
		
		Properties props = new Properties(IMMS.defProps);
		try {
			if(new File("imms.properties").exists()){
				props.load(new FileInputStream("imms.properties"));			 
			}
			props.list(System.out);
			if(new File(keyFilename).exists()){
				new IMMSTester(props).doTest(testFilename,keyFilename);
			} else {
				new IMMSTester(props).doTest(testFilename);
			}
		} catch ( IOException e) {
			throw new RuntimeException(e);		
		}
	}
}
