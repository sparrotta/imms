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
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.util.Span;

public class OpenNlpLemmaAnnotator implements Annotator {

	public static final String ANNOTATION_NAME = "olemma";
	public static final Requirement REQUIREMENT = new Requirement(ANNOTATION_NAME);
	public static final String PROPERTY_MODEL = ANNOTATION_NAME + ".model";
	private LemmatizerME lemmatizer;

	public OpenNlpLemmaAnnotator(Properties props) {
		String modelPath = props.getProperty(PROPERTY_MODEL);
		if (modelPath != null) {
			try {
				this.lemmatizer = new LemmatizerME(new LemmatizerModel(new File(modelPath)));
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
	}

	@Override
	public void annotate(Annotation annotation) {
		List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);

		String[] toks = tokens.stream().map(t -> t.originalText()).collect(Collectors.toList())
				.toArray(new String[] {});
		String[] tags = tokens.stream().map(t -> t.tag()).collect(Collectors.toList()).toArray(new String[] {});
		String[] lemmas = lemmatizer.lemmatize(toks, tags);
		String[] decodedLemmas = lemmatizer.decodeLemmas(toks, lemmas);
		for (int i = 0; i < tokens.size(); i++) {
			tokens.get(i).setLemma(decodedLemmas[i]);
		}
	}

	@Override
	public Set<Requirement> requires() {
		return Collections.unmodifiableSet(new HashSet<Requirement>() {
			{
				add(TOKENIZE_REQUIREMENT);
				add(POS_REQUIREMENT);
			}
		});
	}

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.unmodifiableSet(new HashSet<Requirement>() {
			{
				add(REQUIREMENT);
				add(LEMMA_REQUIREMENT);
			}
		});
	}

}
