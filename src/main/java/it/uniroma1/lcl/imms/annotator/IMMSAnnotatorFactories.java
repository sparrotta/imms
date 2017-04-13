package it.uniroma1.lcl.imms.annotator;

import java.util.Properties;

import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.AnnotatorFactory;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.annotator.feature.LocalCollocationFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.POSFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.SorroundingWordsFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.WordEmbeddingFeatureAnnotator;

public class IMMSAnnotatorFactories {

	public static AnnotatorFactory sorroundingWords(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {
		return new AnnotatorFactory(inputProps,annotatorImplementation) {
	
			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.sorroundingWords(inputProps);
			}

			@Override
			protected String additionalSignature() {				
				StringBuilder sb = new StringBuilder();
				sb.append(SorroundingWordsFeatureAnnotator.PROPERTY_WINDOWSIZE+"=")
				.append(inputProps.getProperty(SorroundingWordsFeatureAnnotator.PROPERTY_WINDOWSIZE, SorroundingWordsFeatureAnnotator.DEFAULT_WINDOWSIZE));
				
				if(inputProps.containsKey(SorroundingWordsFeatureAnnotator.PROPERTY_ADDSTOPWRD)){
					sb.append(SorroundingWordsFeatureAnnotator.PROPERTY_ADDSTOPWRD+"=")
					.append(inputProps.getProperty(SorroundingWordsFeatureAnnotator.PROPERTY_ADDSTOPWRD));
				}
				
				return sb.toString();
			}
		};
	}

	public static AnnotatorFactory sorroundingPOS(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {
		
		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.sorroundingPOS(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return POSFeatureAnnotator.PROPERTY_WINDOWSIZE+"="+inputProps.getProperty(POSFeatureAnnotator.PROPERTY_WINDOWSIZE, POSFeatureAnnotator.DEFAULT_WINDOWSIZE);
			}
		};
	}

	public static AnnotatorFactory localCollocation(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {
		
		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.localCollocations(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return LocalCollocationFeatureAnnotator.PROPERTY_LCOLLOCATIONSET+"="+inputProps.getProperty(LocalCollocationFeatureAnnotator.PROPERTY_LCOLLOCATIONSET, LocalCollocationFeatureAnnotator.DEFAULT_COLLOCATIONS);
			}
		};
	}

	public static AnnotatorFactory wordEmbeddings(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {

		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.wordEmbeddings(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return WordEmbeddingFeatureAnnotator.PROPERTY_WINDOWSIZE+"="+inputProps.getProperty(WordEmbeddingFeatureAnnotator.PROPERTY_WINDOWSIZE, WordEmbeddingFeatureAnnotator.DEFAULT_WINDOWSIZE);
			}
		};
	}

	public static AnnotatorFactory openNlpPosTagger(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {

		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.openNLPPosTagger(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return "";
			}
		};
	}
	
	public static AnnotatorFactory openNlpTokenizer(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {

		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.openNLPTokenizer(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return "";
			}
		};
	}
	
	public static AnnotatorFactory wordnetLemmatizer(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {

		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.wordnetLemmatizer(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return "";
			}
		};
	}

	public static AnnotatorFactory headToken(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {
		
		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.headToken(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return "";
			}
		};
	}

	public static AnnotatorFactory openNlpLemmatizer(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {
		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.openNlpLemma(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return "";
			}
		};
	}
}
