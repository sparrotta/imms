package it.uniroma1.lcl.imms;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.ErasureUtils;
import it.uniroma1.lcl.imms.annotator.feature.Feature;
import net.didion.jwnl.data.POS;

public interface Constants {

	public static String fileSeparator = System.getProperty("file.separator");

	public static final Predicate<String> PREDICATE_IS_WORD = Pattern.compile("[a-zA-Z][\\-a-zA-Z0-9]*").asPredicate();

	public static final String UNKNOWN_SENSE = "UNKNOWN";
	
	public static final String PROPERTY_PREFIX_CUSTOM_ANNOTATOR_IMPLEMENTATION_CLASS = "customAnnotatorsImpl";
	
	public static final String PROPERTY_TASK_HANDLER_CLASS = "task.class";
	public static final String PROPERTY_TASK_RESULT_DIR = "task.resultdir";
	
	public static final String PROPERTY_CLASSIFIER_CLASS = "classifier.class";
	public static final String PROPERTY_CLASSIFIER_MODEL_DIR = "classifier.modeldir";
	public static final String PROPERTY_CLASSIFIER_STAT_DIR = "classifier.statdir";
	public static final String PROPERTY_JWNL_PROP_FILE = "jwnl.propfile";
	
	public static final String PROPERTY_OPEN_NLP = "openNLP";	
	

	public static Map<String,POS> posTagMap = new HashMap<String,POS>();
	
	
	public static class FeaturesAnnotation implements CoreAnnotation<List<Feature>> {
		public Class<List<Feature>> getType() {
			return ErasureUtils.<Class<List<Feature>>> uncheckedCast(List.class);
		}
	}
	
	public static class HeadsAnnotation implements CoreAnnotation<List<CoreMap>> {
		public Class<List<CoreMap>> getType() {
			return ErasureUtils.<Class<List<CoreMap>>> uncheckedCast(List.class);
		}
	}
	public static class HeadTokenAnnotation implements CoreAnnotation<CoreLabel> {
		public Class<CoreLabel> getType() {
			return CoreLabel.class;
		}
	}
	public static class LexicalItemAnnotation implements CoreAnnotation<String> {
		public Class<String> getType() {
			return String.class;
		}
	}
	public static class DocSourceAnnotation implements CoreAnnotation<String> {
		public Class<String> getType() {
			return String.class;
		}
	}
}
