package it.uniroma1.lcl.imms.annotator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.stream.Collectors;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.util.CoreMap;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.util.InvalidFormatException;

public class OpenNlpPosTaggerAnnotator implements Annotator {

	public static final String ANNOTATION_NAME = "opos";
	public static final Requirement OPOS_REQUIREMENT = new Requirement(ANNOTATION_NAME);		
	public static final String PROPERTY_MODEL = ANNOTATION_NAME+".model";
	public static final String PROPERTY_DICTIONARY = ANNOTATION_NAME+".dictionary";
	private POSTaggerME tagger;
	
	public OpenNlpPosTaggerAnnotator(Properties props) {
		String modelPath = props.getProperty(PROPERTY_MODEL);
		String dictPath = props.getProperty(PROPERTY_DICTIONARY);
		if (modelPath != null) {
			try {
				this.tagger = new POSTaggerME(new POSModel(new File(modelPath)));
			} catch (IOException e) {
				throw new RuntimeException(e);
			}			
		}
	}

	@Override
	public void annotate(Annotation annotation) {		
		if (annotation.has(CoreAnnotations.SentencesAnnotation.class)) {
			for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
				List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
			    List<TaggedWord> tagged = null;
			    try {
			    	List<String> words = tokens.stream().map(t->t.originalText()).collect(Collectors.toList());
			    	String[] tags = tagger.tag(words.toArray(new String[words.size()])); 
			    	tagged = new ArrayList<TaggedWord>();			    	
			        for(int i=0; i<tags.length; i++){
			        	tagged.add(new TaggedWord(words.get(i),tags[i]));
			        }
			        
			      } catch (OutOfMemoryError e) {
			        System.err.println("WARNING: Tagging of sentence ran out of memory. " +
			                           "Will ignore and continue: " +
			                           Sentence.listToString(tokens));
			      }

			    if (tagged != null) {
			      for (int i = 0, sz = tokens.size(); i < sz; i++) {
			        tokens.get(i).set(CoreAnnotations.PartOfSpeechAnnotation.class, tagged.get(i).tag());
			      }
			    } else {
			      for (CoreLabel token : tokens) {
			        token.set(CoreAnnotations.PartOfSpeechAnnotation.class, "X");
			      }
			    }			    
		    }
		}
		

	}

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.unmodifiableSet(new HashSet<Requirement>() {{
	        add(POS_REQUIREMENT);
	        add(OPOS_REQUIREMENT);
	      }});
	}

	@Override
	public Set<Requirement> requires() {		
		return Collections.unmodifiableSet(new HashSet<Requirement>() {{
	        add(TOKENIZE_REQUIREMENT);
	        add(SSPLIT_REQUIREMENT);
	      }});
	}

}
