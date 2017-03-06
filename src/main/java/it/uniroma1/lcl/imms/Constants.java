package it.uniroma1.lcl.imms;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator.Requirement;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.ErasureUtils;
import it.uniroma1.lcl.imms.annotator.feature.Feature;

import java.util.List;
import java.util.function.Predicate;
import java.util.regex.Pattern;

public interface Constants {

	public static String fileSeparator = System.getProperty("file.separator");

	public static final Predicate<String> PREDICATE_IS_WORD = Pattern.compile("[a-zA-Z][_a-zA-Z0-9]*").asPredicate();

	public static final String UNKNOWN_SENSE = "UNKNOWN";
	
	public static final String CLASSIFIER_IMMS_LIBLINEAR = "liblinear";
	
	public static final String ANNOTATOR_IMMS_HEADTOKEN = "head";
	public static final String ANNOTATOR_FEAT_IMMS_SRNDWORDS = "feat_sorround";
	public static final String ANNOTATOR_FEAT_IMMS_POS = "feat_pos";
	public static final String ANNOTATOR_FEAT_IMMS_LCOLLOCATION = "feat_lcollocation";
	public static final String ANNOTATOR_FEAT_IMMS_WORDEMBED = "feat_wordembed";

	public static final Requirement REQUIREMENT_ANNOTATOR_IMMS_HEADTOKEN = new Requirement(ANNOTATOR_IMMS_HEADTOKEN);
	public static final Requirement REQUIREMENT_ANNOTATOR_FEAT_IMMS_SRNDWORDS = new Requirement(ANNOTATOR_FEAT_IMMS_SRNDWORDS);
	public static final Requirement REQUIREMENT_ANNOTATOR_FEAT_IMMS_POS = new Requirement(ANNOTATOR_FEAT_IMMS_POS);
	public static final Requirement REQUIREMENT_ANNOTATOR_FEAT_IMMS_LCOLLOCATION = new Requirement(ANNOTATOR_FEAT_IMMS_LCOLLOCATION);
	public static final Requirement REQUIREMENT_ANNOTATOR_FEAT_IMMS_WORDEMBED = new Requirement(ANNOTATOR_FEAT_IMMS_LCOLLOCATION);
	
	public static final String PROPERTY_PREFIX_CUSTOM_ANNOTATOR_IMPLEMENTATION_CLASS = "customAnnotatorsImpl";
	
	public static final String PROPERTY_TASK_HANDLER_CLASS = "taskClass";
	public static final String PROPERTY_TASK_RESULT_DIR = "task_resultdir";
	
	public static final String PROPERTY_CLASSIFIER_CLASS = "classifierClass";
	public static final String PROPERTY_CLASSIFIER_MODEL_DIR = "classifier_modeldir";
	public static final String PROPERTY_CLASSIFIER_STAT_DIR = "classifier_statdir";
	
	public static final String PROPERTY_OPEN_NLP = "openNLP";

	public static final String PROPERTY_IMMS_SRNDWORDS_WINDOWSIZE = ANNOTATOR_FEAT_IMMS_SRNDWORDS+".windowsize";
	public static final String PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD = ANNOTATOR_FEAT_IMMS_SRNDWORDS+".addStopWords";
	public static final String PROPERTY_IMMS_POS_WINDOWSIZE = ANNOTATOR_FEAT_IMMS_POS+".windowsize";
	public static final String PROPERTY_IMMS_LCOLLOCATIONSET = ANNOTATOR_FEAT_IMMS_LCOLLOCATION+".set";
	public static final String PROPERTY_IMMS_WORDEMBED_WINDOWSIZE = ANNOTATOR_FEAT_IMMS_WORDEMBED + ".windowsize";
	public static final String PROPERTY_IMMS_WORDEMBED_FILE = ANNOTATOR_FEAT_IMMS_WORDEMBED + ".file";
	public static final String PROPERTY_IMMS_WORDEMBED_STRATEGY = ANNOTATOR_FEAT_IMMS_WORDEMBED + ".strategy";
	public static final String PROPERTY_IMMS_LIBLINEAR_BIAS = CLASSIFIER_IMMS_LIBLINEAR + ".bias";

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
