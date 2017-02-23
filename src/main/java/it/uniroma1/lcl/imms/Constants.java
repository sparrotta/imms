package it.uniroma1.lcl.imms;

import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotator.Requirement;
import edu.stanford.nlp.util.ErasureUtils;
import it.uniroma1.lcl.imms.feature.Feature;

public interface Constants {

	Predicate<String> PREDICATE_IS_WORD = Pattern.compile("[a-zA-Z][_a-zA-Z0-9]*").asPredicate();

	public static final String UNKNOWN_SENSE = "UNKNOWN";
	
	public static final String IMMS_SRNDWORDS = "feat_sorround";
	public static final Requirement IMMS_SRNDWORDS_REQUIREMENT = new Requirement(IMMS_SRNDWORDS);
	public static final String PROPERTY_IMMS_SRNDWORDS_WINDOWSIZE = IMMS_SRNDWORDS+".windowsize";
	public static final String PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD = IMMS_SRNDWORDS+".addStopWords";
	
	public static final String IMMS_POS = "feat_pos";
	public static final Requirement IMMS_POS_REQUIREMENT = new Requirement(IMMS_POS);		
	public static final String PROPERTY_IMMS_POS_WINDOWSIZE = IMMS_POS+".windowsize";
	
	public static final String IMMS_LCOLLOCATION = "feat_lcollocation";
	public static final Requirement IMMS_LCOLLOCATION_REQUIREMENT = new Requirement(IMMS_POS);
	public static final String PROPERTY_IMMS_LCOLLOCATIONSET = IMMS_LCOLLOCATION+".set";

	
	public static class FeaturesAnnotation implements CoreAnnotation<List<Feature>> {
		public Class<List<Feature>> getType() {
			return ErasureUtils.<Class<List<Feature>>> uncheckedCast(List.class);
		}
	}
	
	public static class HeadAnnotation implements CoreAnnotation<CoreLabel> {
		public Class<CoreLabel> getType() {
			return CoreLabel.class;
		}
	}
	public static class LexicalElementAnnotation implements CoreAnnotation<String> {
		public Class<String> getType() {
			return String.class;
		}
	}
}
