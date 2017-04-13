package it.uniroma1.lcl.imms.annotator;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import it.uniroma1.lcl.imms.Constants;
import net.didion.jwnl.JWNL;
import net.didion.jwnl.JWNLException;
import net.didion.jwnl.data.POS;
import net.didion.jwnl.dictionary.Dictionary;
import net.didion.jwnl.dictionary.MorphologicalProcessor;
import opennlp.tools.postag.POSTaggerME;

public class WordNetLemmaAnnotator implements Annotator {

	public static final String ANNOTATION_NAME = "wnlemma";
	public static final Requirement REQUIREMENT = new Requirement(ANNOTATION_NAME);			
	
	private POSTaggerME tagger;
	private Dictionary dictionary;
	private MorphologicalProcessor processor;
	
	public WordNetLemmaAnnotator(Properties props) {	
		String propertyFilename= props.getProperty(Constants.PROPERTY_JWNL_PROP_FILE);
		try {
			JWNL.initialize(new FileInputStream(propertyFilename));
			dictionary = Dictionary.getInstance();
			processor = dictionary.getMorphologicalProcessor();
		} catch (FileNotFoundException | JWNLException e) {			
			throw new RuntimeException(e);
		}
		
	}

	@Override
	public void annotate(Annotation annotation) {
		List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
		for(CoreLabel token : tokens){
			String tokenLemma = token.originalText().trim();
			String word = token.word();
			POS posTag = Constants.posTagMap.get(token.tag().toLowerCase().substring(0, 1));
			if(posTag!=null && word!=null){
				try {									
					List<String> indexWords = processor.lookupAllBaseForms(posTag, word);
					if (indexWords.size() > 0) {
						tokenLemma = indexWords.get(0);
						for (String lemma:indexWords) {
							if (lemma.equals(word)) {
								tokenLemma = lemma;
								break;
							}
						}							
					}
				} catch (JWNLException e) {
					throw new RuntimeException(e);
				}
			}
			
			token.setLemma(tokenLemma);
		}				
	}

	

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.unmodifiableSet(new HashSet<Requirement>() {{	        
	        add(REQUIREMENT);
	        add(LEMMA_REQUIREMENT);
	      }});
	}

	@Override
	public Set<Requirement> requires() {		
		return Collections.unmodifiableSet(new HashSet<Requirement>() {{
	        add(TOKENIZE_REQUIREMENT);	        
	        add(POS_REQUIREMENT);	        
	      }});
	}

}
