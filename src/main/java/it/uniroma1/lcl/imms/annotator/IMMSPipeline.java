package it.uniroma1.lcl.imms.annotator;

import java.lang.reflect.InvocationTargetException;
import java.util.Properties;

import edu.stanford.nlp.pipeline.AnnotatorImplementations;
import edu.stanford.nlp.pipeline.AnnotatorPool;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.annotator.feature.LocalCollocationFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.POSFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.SorroundingWordsFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.WordEmbeddingFeatureAnnotator;
import it.uniroma1.lcl.imms.classifiers.Classifier;
import it.uniroma1.lcl.imms.classifiers.LibLinearClassifier;
import it.uniroma1.lcl.imms.task.ITaskHandler;
import it.uniroma1.lcl.imms.task.impl.SensEval2LexicalSampleTask;

public class IMMSPipeline extends StanfordCoreNLP {

	public IMMSPipeline(Properties props) {
		super(props);
	}

	@Override
	protected AnnotatorImplementations getAnnotatorImplementations() {
		return new IMMSAnnotatorImplementations();
	}

	@Override
	protected synchronized AnnotatorPool getDefaultAnnotatorPool(Properties inputProps,
			AnnotatorImplementations annotatorImplementation) {
		AnnotatorPool superPool = super.getDefaultAnnotatorPool(inputProps, annotatorImplementation);

		superPool.register(HeadTokenAnnotator.ANNOTATION_NAME,
				IMMSAnnotatorFactories.headToken(inputProps, (IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(SorroundingWordsFeatureAnnotator.ANNOTATION_NAME, IMMSAnnotatorFactories.sorroundingWords(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(POSFeatureAnnotator.ANNOTATION_NAME, IMMSAnnotatorFactories.sorroundingPOS(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(LocalCollocationFeatureAnnotator.ANNOTATION_NAME, IMMSAnnotatorFactories
				.localCollocation(inputProps, (IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(WordEmbeddingFeatureAnnotator.ANNOTATION_NAME, IMMSAnnotatorFactories.wordEmbeddings(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(OpenNlpPosTaggerAnnotator.ANNOTATION_NAME, IMMSAnnotatorFactories
				.openNlpPosTagger(inputProps, (IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(OpenNlpTokenizeAnnotator.ANNOTATION_NAME, IMMSAnnotatorFactories.openNlpTokenizer(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(WordNetLemmaAnnotator.ANNOTATION_NAME, IMMSAnnotatorFactories.wordnetLemmatizer(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(OpenNlpLemmaAnnotator.ANNOTATION_NAME, IMMSAnnotatorFactories.openNlpLemmatizer(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));

		if (Boolean.parseBoolean(inputProps.getProperty("openNLP", "false"))) {
			superPool.register(STANFORD_POS, IMMSAnnotatorFactories.openNlpPosTagger(inputProps,
					(IMMSAnnotatorImplementations) annotatorImplementation));
			superPool.register(STANFORD_TOKENIZE, IMMSAnnotatorFactories.openNlpTokenizer(inputProps,
					(IMMSAnnotatorImplementations) annotatorImplementation));
			superPool.register(STANFORD_LEMMA, IMMSAnnotatorFactories
					.openNlpLemmatizer(inputProps, (IMMSAnnotatorImplementations) annotatorImplementation));
		}

		return superPool;
	}

	public ITaskHandler getTaskHandler() {
		String className = this.getProperties().getProperty(Constants.PROPERTY_TASK_HANDLER_CLASS,
				SensEval2LexicalSampleTask.class.getName());
		try {
			return (ITaskHandler) Class.forName(className).getConstructor(Properties.class).newInstance(this.getProperties());
		} catch (InstantiationException | IllegalAccessException | ClassNotFoundException | IllegalArgumentException | InvocationTargetException | NoSuchMethodException | SecurityException e) {
			throw new RuntimeException(e);
		}
	}

	public Classifier getClassifier() {
		String className = this.getProperties().getProperty(Constants.PROPERTY_CLASSIFIER_CLASS,
				LibLinearClassifier.class.getName());
		try {
			return (Classifier) Class.forName(className).getConstructor(Properties.class)
					.newInstance(this.getProperties());
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
				| NoSuchMethodException | SecurityException | ClassNotFoundException e) {
			throw new RuntimeException(e);
		}

	}

}
