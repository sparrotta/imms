package it.uniroma1.lcl.imms;

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
import it.uniroma1.lcl.imms.classifiers.Classifier;
import it.uniroma1.lcl.imms.classifiers.LibLinearClassifier;
import it.uniroma1.lcl.imms.corpus.ICorpusReader;
import it.uniroma1.lcl.imms.corpus.impl.SensEvalLexicalSampleCorpus;

public class IMMSTrainer {

	
	private IMMSPipeline pipeline;
	private Properties properties;

	public IMMSTrainer(Properties props) {
		this.properties = props;
		this.pipeline = new IMMSPipeline(props);		
	}

	void doTrain(String trainFile) throws FileNotFoundException, IOException {
		ICorpusReader cr = this.pipeline.getCorpusReader();
		cr.loadCorpus(trainFile);		
		doTrain(cr);				
	}
	void doTrain(String trainFile, String keyFile) throws FileNotFoundException, IOException {
		ICorpusReader cr = this.pipeline.getCorpusReader();
		cr.loadCorpus(trainFile);
		cr.loadAnswers(keyFile);
		doTrain(cr);				
	}
	void doTrain(ICorpusReader corpusReader) {				
		Iterator<Annotation> it = corpusReader.iterator();
		Classifier classifier = pipeline.getClassifier();
		 
		while (it.hasNext()) {
			Annotation text = it.next();
			pipeline.annotate(text);						
			classifier.add(text);			
		}		
		classifier.train();
		try {
			classifier.write(properties.getProperty("outdir"));
		} catch (IOException e) {			
			throw new RuntimeException(e);
		}
	}

	
	public static void main(String[] args) {

		Options options = new Options();

		// Option input = new Option("i", "input", true, "input file path");
		// input.setRequired(true);
		// options.addOption(input);
		//
		// Option output = new Option("o", "output", true, "output file");
		// output.setRequired(true);
		// options.addOption(output);

		CommandLineParser parser = new DefaultParser();
		HelpFormatter formatter = new HelpFormatter();
		CommandLine cmd;

		try {
			cmd = parser.parse(options, args);
			if (cmd.getArgList().size() < 3) {
				throw new ParseException("Missing file or output dir");
			}
		} catch (ParseException e) {
			System.out.println(e.getMessage());
			formatter.printHelp("imms train.xml train.key saveDir", options);
			System.exit(1);
			return;
		}

		String[] fileArgs = cmd.getArgs();
		String trainXmlFilename = fileArgs[0];
		String keyFilename = fileArgs[1];
		String outputDir = fileArgs[2];

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
		Properties props = new Properties();
		props.setProperty("outdir", "out");
		props.setProperty("feat_wordembed.file", "wordvectors.txt");
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, feat_sorround, feat_pos, feat_lcollocation, feat_wordembed");		
		try {
			new IMMSTrainer(props).doTrain(trainXmlFilename,keyFilename);
		} catch ( IOException e) {
			throw new RuntimeException(e);
		}
		
	}
}
