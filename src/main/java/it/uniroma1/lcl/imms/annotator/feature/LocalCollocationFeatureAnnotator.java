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

public class LocalCollocationFeatureAnnotator implements Annotator {

	public static final String ANNOTATION_NAME = "feat_lcollocation";
	public static final String FEATURE_PREFIX = "LOC_";
	public static final Requirement REQUIREMENT = new Requirement(ANNOTATION_NAME);

	
	public static final String PROPERTY_LCOLLOCATIONSET = ANNOTATION_NAME+".set";
	
	public static final String DEFAULT_COLLOCATIONS = "-2:-2,-1:-1,1:1,2:2,-2:-1,-1:1,1:2,-3:-1,-2:1,-1:2,1:3";
	String collocations;
	
	public LocalCollocationFeatureAnnotator(Properties properties) {
		collocations = properties.getProperty(PROPERTY_LCOLLOCATIONSET, DEFAULT_COLLOCATIONS);
	}

	@Override
	public void annotate(Annotation annotation) {		
		for(CoreMap head : annotation.get(HeadsAnnotation.class)){
			head.get(Constants.FeaturesAnnotation.class).addAll(featurize(head.get(HeadTokenAnnotation.class), annotation));
		}
	}
	
	List<Feature> featurize(CoreLabel head, Annotation annotation){
		List<CoreLabel> tokens = null;				 		
		List<Feature> features = new ArrayList<Feature>();				
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		for (CoreMap sentence : sentences) {
			if(sentence.get(SentenceIndexAnnotation.class)==head.sentIndex()){			
				tokens = sentence.get(TokensAnnotation.class);
				break;
			}			
		}
		if(tokens!=null){			
			Integer headIndex=head.index()-tokens.get(0).index();
			
			for(String collIndexPair:collocations.split(",")){
				String [] pair = collIndexPair.split(":");
				int start = Integer.parseInt(pair[0]);
				int end = Integer.parseInt(pair[1]);
				String key = FEATURE_PREFIX+start+":"+end;
				for(int i=headIndex+start;i<=headIndex+end;i++){
					if(i==headIndex){
						continue;
					} else if(i>-1 && i<tokens.size()){
						key+="_"+tokens.get(i).lemma();
					} else {
						key+="_^";
					}
				}
				features.add(new Feature<Boolean>(key,true));
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
		return Collections.unmodifiableSet(new ArraySet<>(TOKENIZE_REQUIREMENT, SSPLIT_REQUIREMENT,LEMMA_REQUIREMENT,HeadTokenAnnotator.REQUIREMENT));
	}

}
