package it.uniroma1.lcl.imms;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator.Requirement;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.ErasureUtils;
import it.uniroma1.lcl.imms.feature.Feature;

import java.util.List;
import java.util.function.Predicate;
import java.util.regex.Pattern;

public interface Constants {

	
	Predicate<String> PREDICATE_IS_WORD = Pattern.compile("[a-zA-Z][_a-zA-Z0-9]*").asPredicate();

	public static final String CUSTOM_ANNOTATOR_IMPLEMENTATION_CLASS = "customAnnotatorsImpl";
	public static final String IMMS_CORPUS_READER_CLASS = "corpusReader";
	public static final String IMMS_CLASSIFIER_CLASS = "classifier";
	
	public static final String UNKNOWN_SENSE = "UNKNOWN";
	
	public static final String IMMS_SRNDWORDS = "feat_sorround";
	public static final Requirement IMMS_SRNDWORDS_REQUIREMENT = new Requirement(IMMS_SRNDWORDS);
	public static final String PROPERTY_IMMS_SRNDWORDS_WINDOWSIZE = IMMS_SRNDWORDS+".windowsize";
	public static final String PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD = IMMS_SRNDWORDS+".addStopWords";
	
	public static final String IMMS_POS = "feat_pos";
	public static final Requirement IMMS_POS_REQUIREMENT = new Requirement(IMMS_POS);		
	public static final String PROPERTY_IMMS_POS_WINDOWSIZE = IMMS_POS+".windowsize";
	
	public static final String IMMS_LCOLLOCATION = "feat_lcollocation";
	public static final Requirement IMMS_LCOLLOCATION_REQUIREMENT = new Requirement(IMMS_LCOLLOCATION);
	public static final String PROPERTY_IMMS_LCOLLOCATIONSET = IMMS_LCOLLOCATION+".set";

	public static final String IMMS_WORDEMBED = "feat_wordembed";
	public static final Requirement IMMS_WORDEMBED_REQUIREMENT = new Requirement(IMMS_LCOLLOCATION);
	public static final String PROPERTY_IMMS_WORDEMBED_WINDOWSIZE = IMMS_WORDEMBED + ".windowsize";
	public static final String PROPERTY_IMMS_WORDEMBED_FILE = IMMS_WORDEMBED + ".file";
	public static final String PROPERTY_IMMS_WORDEMBED_STRATEGY = IMMS_WORDEMBED + ".strategy";
	
	public static final String PROPERTY_IMMS_LIBLINEAR = "liblinear";
	public static final String PROPERTY_IMMS_LIBLINEAR_BIAS = PROPERTY_IMMS_LIBLINEAR + ".bias";

	public static class FeaturesAnnotation implements CoreAnnotation<List<Feature>> {
		public Class<List<Feature>> getType() {
			return ErasureUtils.<Class<List<Feature>>> uncheckedCast(List.class);
		}
	}
	
	public static class HeadsAnnotation implements CoreAnnotation<List<CoreLabel>> {
		public Class<List<CoreLabel>> getType() {
			return ErasureUtils.<Class<List<CoreLabel>>> uncheckedCast(List.class);
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
