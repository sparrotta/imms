package it.uniroma1.lcl.imms;

import java.util.Arrays;
import java.util.Properties;

public class IMMS {
	public static Properties defProps = new Properties();
	
	static {				
		defProps.setProperty(Constants.PROPERTY_CLASSIFIER_MODEL_DIR, "out");
		defProps.setProperty(Constants.PROPERTY_CLASSIFIER_STAT_DIR, "out");
		defProps.setProperty(Constants.PROPERTY_TASK_RESULT_DIR, "out");
		defProps.setProperty(Constants.PROPERTY_IMMS_WORDEMBED_FILE, "wordvectors.txt");
		defProps.setProperty(Constants.PROPERTY_IMMS_WORDEMBED_STRATEGY, "concatenation");
		defProps.setProperty(Constants.PROPERTY_IMMS_WORDEMBED_SIGMA, "0.1");
		defProps.setProperty("annotators", "tokenize, ssplit, pos, lemma, "+Constants.ANNOTATOR_IMMS_HEADTOKEN+", "+Constants.ANNOTATOR_FEAT_IMMS_SRNDWORDS+", "+Constants.ANNOTATOR_FEAT_IMMS_POS+", "+Constants.ANNOTATOR_FEAT_IMMS_LCOLLOCATION+", "+Constants.ANNOTATOR_FEAT_IMMS_WORDEMBED);		
	}
	
	public static void main(String[] args) {
		String trainOpt = "-train";
		String testOpt = "-test";
		String generalOptions = "Usage: ["+trainOpt+"|"+testOpt+"]\n";
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
