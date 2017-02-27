package it.uniroma1.lcl.imms;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Map.Entry;

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

public class IMMSTester {

	
	private IMMSPipeline pipeline;
	private Properties properties;

	public IMMSTester(Properties props) {
		this.properties = props;
		this.pipeline = new IMMSPipeline(props);		
	}

	

	void doTest(String testFile) throws FileNotFoundException, IOException {
		ICorpusReader cr = this.pipeline.getCorpusReader();
		cr.loadCorpus(testFile);
		doTest(cr);				
	}
	void doTest(String testFile, String keyFile) throws FileNotFoundException, IOException {
		ICorpusReader cr = this.pipeline.getCorpusReader();
		cr.loadCorpus(testFile);
		cr.loadAnswers(keyFile);
		doTest(cr);				
	}
	void doTest(ICorpusReader corpusReader) {				
		Iterator<Annotation> it = corpusReader.iterator();
		Classifier classifier = pipeline.getClassifier();
		classifier.setModelDir(properties.getProperty("modeldir"));
		classifier.setStatDir(properties.getProperty("statdir"));		
		
		while (it.hasNext()) {
			Annotation text = it.next();
			pipeline.annotate(text);
			classifier.add(text);
		}
		
		Map<String,List<String>> lexElemAnswersMap = classifier.test();		
		for(Entry<String, List<String>>entry : lexElemAnswersMap.entrySet()){
			String lexElem = entry.getKey();
			List<String> answers = entry.getValue();			
			for(int i=0; i<answers.size();i++){
				System.out.println(classifier.src(lexElem,i)+"\t"+classifier.id(lexElem,i)+"\t"+answers.get(i));
			}
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
		String testPath = fileArgs[0];
		String modelDir = fileArgs[1];
		String statDir = fileArgs[2];
//		String saveDir = fileArgs[3];

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
		Properties props = new Properties();
		props.setProperty("modeldir", "out");
		props.setProperty("statdir", "out");
		props.setProperty("feat_wordembed.file", "wordvectors.txt");
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, feat_sorround, feat_pos, feat_lcollocation, feat_wordembed");		
		
		try {			 
			new IMMSTester(props).doTest(testPath);
		} catch ( IOException e) {
			throw new RuntimeException(e);		
		}
	}
}
