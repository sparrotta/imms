package it.uniroma1.lcl.imms;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;

import javax.xml.stream.XMLStreamException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.Index;
import it.uniroma1.lcl.imms.annotator.IMMSPipeline;
import it.uniroma1.lcl.imms.classifiers.Classifier;
import it.uniroma1.lcl.imms.classifiers.IMMSDataset;
import it.uniroma1.lcl.imms.classifiers.LibLinearClassifier;
import it.uniroma1.lcl.imms.task.ITaskHandler;
import it.uniroma1.lcl.imms.task.impl.SensEval2LexicalSampleTask;

public class IMMSTrainer {

	
	private IMMSPipeline pipeline;
	private Properties properties;

	public IMMSTrainer(Properties props) {
		this.properties = props;
		this.pipeline = new IMMSPipeline(props);		
	}

	void doTrain(String trainFile) throws FileNotFoundException, IOException {
		ITaskHandler cr = this.pipeline.getTaskHandler();
		cr.loadCorpus(trainFile);		
		doTrain(cr);				
	}
	void doTrain(String trainFile, String keyFile) throws FileNotFoundException, IOException {
		ITaskHandler cr = this.pipeline.getTaskHandler();
		cr.loadCorpus(trainFile);
		cr.loadAnswers(keyFile);
		doTrain(cr);				
	}
	void doTrain(ITaskHandler corpusReader) {				
		Iterator<Annotation> it = corpusReader.iterator();
		Classifier classifier = pipeline.getClassifier();
		 int cnt=0;
		while (it.hasNext()) {
			Annotation text = it.next();
			pipeline.annotate(text);						
			classifier.add(text);			
		}		
		classifier.train();
	
		try {
			classifier.write(properties.getProperty(Constants.PROPERTY_CLASSIFIER_MODEL_DIR),properties.getProperty(Constants.PROPERTY_CLASSIFIER_STAT_DIR));
		} catch (IOException e) {			
			throw new RuntimeException(e);
		}
	}

	
	public static void main(String[] args) {	
		
		String trainFilename = args[0];
		String keyFilename = trainFilename+".key";

		//Just as a reminder of original ims command line
		//TODO porting from command line into properties
		String generalOptions = "Usage: train.xml train.key saveDir\n"
				+ "\t-i class name of Instance Extractor(default sg.edu.nus.comp.nlp.ims.instance.CInstanceExtractor)\n"
				+ "\t-f class name of Feature Extractor(default sg.edu.nus.comp.nlp.ims.feature.CMixedFeatureExtractor)\n"
				+ "\t-c class name of Corpus(default sg.edu.nus.comp.nlp.ims.corpus.CLexicalCorpus)\n"
				+ "\t-t class name of Trainer(default sg.edu.nus.comp.nlp.ims.classifiers.CLibLinearTrainer)\n"
				+ "\t-m class name of Model Writer(default sg.edu.nus.comp.nlp.ims.io.CModelWriter)\n"
				+ "\t-algorithm svm(default) or naivebayes\n"
				+ "\t-s2 cut off for surrounding word(default 0)\n"
				+ "\t-c2 cut off for collocation(default 0)\n"
				+ "\t-p2 cut off for pos(default 0)\n"
				+ "\t-split 1/0 whether the corpus is sentence splitted(default 0)\n"
				+ "\t-ssm path of sentence splitter model\n"
				+ "\t-token 1/0 whether the corpus is tokenized(default 0)\n"
				+ "\t-pos 1/0 whether the pos tag is provided in corpus(default 0)\n"
				+ "\t-ptm path of pos tagger model\n"
				+ "\t-dict path of dictionary for opennlp POS tagger\n"
				+ "\t-tagdict path of tagdict for opennlp POS tagger\n"
				+ "\t-lemma 1/0 whether the lemma is provided in the corpus(default 0)\n"
				+ "\t-prop path of prop.xml for JWNL\n"
				+ "\t-type type of train.xml\n"
				+ "\t\tdirectory train all xml files under directory trainPath\n"
				+ "\t\tlist train all xml files listed in file trainPath\n"
				+ "\t\tfile(default) train file trainPath\n";
		
		
		Properties props = new Properties(IMMS.defProps);
		try {
			if(new File("imms.properties").exists()){
				props.load(new FileInputStream("imms.properties"));			 
			}
			props.list(System.out);
			if(new File(keyFilename).exists()){
				new IMMSTrainer(props).doTrain(trainFilename,keyFilename);
			} else {
				new IMMSTrainer(props).doTrain(trainFilename);
			}
		} catch ( IOException e) {
			throw new RuntimeException(e);
		}
		
	}
}
