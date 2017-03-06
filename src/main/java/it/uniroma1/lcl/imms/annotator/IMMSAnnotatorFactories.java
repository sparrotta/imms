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
				sb.append(Constants.PROPERTY_IMMS_SRNDWORDS_WINDOWSIZE+"=")
				.append(inputProps.getProperty(Constants.PROPERTY_IMMS_SRNDWORDS_WINDOWSIZE, SorroundingWordsFeatureAnnotator.DEFAULT_WINDOWSIZE));
				
				if(inputProps.containsKey(Constants.PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD)){
					sb.append(Constants.PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD+"=")
					.append(inputProps.getProperty(Constants.PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD));
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
				return Constants.PROPERTY_IMMS_POS_WINDOWSIZE+"="+inputProps.getProperty(Constants.PROPERTY_IMMS_POS_WINDOWSIZE, POSFeatureAnnotator.DEFAULT_WINDOWSIZE);
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
				return Constants.PROPERTY_IMMS_LCOLLOCATIONSET+"="+inputProps.getProperty(Constants.PROPERTY_IMMS_LCOLLOCATIONSET, LocalCollocationFeatureAnnotator.DEFAULT_COLLOCATIONS);
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
				return Constants.PROPERTY_IMMS_WORDEMBED_WINDOWSIZE+"="+inputProps.getProperty(Constants.PROPERTY_IMMS_WORDEMBED_WINDOWSIZE, WordEmbeddingFeatureAnnotator.DEFAULT_WINDOWSIZE);
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
}
