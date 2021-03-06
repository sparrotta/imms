package it.uniroma1.lcl.imms.annotator.feature;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentenceIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokenBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokenEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.Annotator.Requirement;
import edu.stanford.nlp.util.ArraySet;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.HeadTokenAnnotation;
import it.uniroma1.lcl.imms.Constants.HeadsAnnotation;
import it.uniroma1.lcl.imms.annotator.HeadTokenAnnotator;

public class POSFeatureAnnotator implements Annotator {

	public static final String ANNOTATION_NAME = "feat_pos";
	public static final String FEATURE_PREFIX = "POS_";
	public static final Requirement REQUIREMENT = new Requirement(ANNOTATION_NAME);

	public static final String PROPERTY_WINDOWSIZE = ANNOTATION_NAME+ ".windowsize";
	public static final String PROPERTY_PUNCTUATION = ANNOTATION_NAME+ ".punctuation";

	public static final String DEFAULT_WINDOWSIZE = "3";
	public static final String DEFAULT_PUNCTUATION = "false";
	
	Integer windowSize;
	boolean punctuation;
	
	public POSFeatureAnnotator(Properties properties) {
		windowSize = Integer.valueOf(properties.getProperty(PROPERTY_WINDOWSIZE,DEFAULT_WINDOWSIZE));
		punctuation = Boolean.valueOf(properties.getProperty(PROPERTY_PUNCTUATION,DEFAULT_PUNCTUATION));
	}

	@Override
	public void annotate(Annotation annotation) {
		for(CoreMap head : annotation.get(HeadsAnnotation.class)){
			head.get(Constants.FeaturesAnnotation.class).addAll(featurize(head.get(HeadTokenAnnotation.class), annotation));
		}
	}

	private List<Feature> featurize(CoreLabel head, Annotation annotation) {
		List<String> before = new ArrayList<String>();
		List<String> after = new ArrayList<String>();
		List<Feature> features = new ArrayList<Feature>();		
		
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		List<CoreLabel> tokens = null;
		for (CoreMap sentence : sentences) {
			if(sentence.get(SentenceIndexAnnotation.class)==head.sentIndex()){			
				tokens = sentence.get(TokensAnnotation.class);
				break;
			}								
		}
		if(tokens!=null){
			Integer headIndex=head.index()-tokens.get(0).index();
			features.add(new Feature<Boolean>(FEATURE_PREFIX+"0_"+head.tag(),true));
			int position = 0;
			for(int i=headIndex-1; i >= 0 && position < windowSize; i--){
				CoreLabel token = tokens.get(i);
				if(punctuation || Constants.PREDICATE_IS_WORD.test(token.lemma())){
					features.add(new Feature<Boolean>(FEATURE_PREFIX+"-" + (++position) + "_" + token.tag(), true));
				}					
			}
			position = 0;
			for(int i=headIndex+1; i < tokens.size() && position < windowSize; i++){
				CoreLabel token = tokens.get(i);
				if(punctuation || Constants.PREDICATE_IS_WORD.test(token.lemma())){
					features.add(new Feature<Boolean>(FEATURE_PREFIX+ (++position) + "_" + token.tag(), true));
				}					
			}
		}
		return features;	
		
	}

	
	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.singleton(REQUIREMENT);
	}

	@Override
	public Set<Requirement> requires() {
		return Collections.unmodifiableSet(new ArraySet<>(TOKENIZE_REQUIREMENT, SSPLIT_REQUIREMENT,POS_REQUIREMENT,HeadTokenAnnotator.REQUIREMENT));
	}

}
