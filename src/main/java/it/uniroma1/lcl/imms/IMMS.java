package it.uniroma1.lcl.imms;

import java.util.Arrays;
import java.util.Properties;

import it.uniroma1.lcl.imms.annotator.HeadTokenAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.LocalCollocationFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.POSFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.SorroundingWordsFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.WordEmbeddingFeatureAnnotator;
import net.didion.jwnl.data.POS;

public class IMMS {
	public static Properties defProps = new Properties();
	
	static {				
		defProps.setProperty(Constants.PROPERTY_CLASSIFIER_MODEL_DIR, "models");
		defProps.setProperty(Constants.PROPERTY_CLASSIFIER_STAT_DIR, "stats");
		defProps.setProperty(Constants.PROPERTY_TASK_RESULT_DIR, "results");
		defProps.setProperty(WordEmbeddingFeatureAnnotator.PROPERTY_FILE, "wordvectors.txt");
		defProps.setProperty(WordEmbeddingFeatureAnnotator.PROPERTY_STRATEGY, "concatenation");
		defProps.setProperty(WordEmbeddingFeatureAnnotator.PROPERTY_SIGMA, "0");
		defProps.setProperty("annotators", "tokenize, ssplit, pos, lemma, "+HeadTokenAnnotator.ANNOTATION_NAME+", "+SorroundingWordsFeatureAnnotator.ANNOTATION_NAME+", "+POSFeatureAnnotator.ANNOTATION_NAME+", "+LocalCollocationFeatureAnnotator.ANNOTATION_NAME+", "+WordEmbeddingFeatureAnnotator.ANNOTATION_NAME);		
	}
	
	static {
		Constants.posTagMap.put("n", POS.NOUN);
		Constants.posTagMap.put("v", POS.VERB);
		Constants.posTagMap.put("j", POS.ADJECTIVE);
		Constants.posTagMap.put("a", POS.ADJECTIVE);
		Constants.posTagMap.put("r", POS.ADVERB);
	}
	
	public static void main(String[] args) {
		String trainOpt = "-train";
		String testOpt = "-test";
		String generalOptions = "Usage: IMMS ["+trainOpt+"|"+testOpt+"]\n";
		boolean train=false,test=false;
		if(args.length > 0){
			train = trainOpt.equals(args[0]);
			test = testOpt.equals(args[0]);

		}	
		if(!train && !test){
			System.out.println(generalOptions);
		} else if(train){
			IMMSTrainer.main(Arrays.copyOfRange(args, 1, args.length));
		} else if(test){
			IMMSTester.main(Arrays.copyOfRange(args, 1, args.length));
		}
		
		

	}

}
