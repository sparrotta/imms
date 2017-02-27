package it.uniroma1.lcl.imms;

import java.lang.reflect.InvocationTargetException;
import java.util.Properties;

import edu.stanford.nlp.pipeline.AnnotatorImplementations;
import edu.stanford.nlp.pipeline.AnnotatorPool;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import it.uniroma1.lcl.imms.classifiers.Classifier;
import it.uniroma1.lcl.imms.classifiers.LibLinearClassifier;
import it.uniroma1.lcl.imms.corpus.ICorpusReader;
import it.uniroma1.lcl.imms.corpus.impl.SensEvalLexicalSampleCorpus;

public class IMMSPipeline extends StanfordCoreNLP {

	public IMMSPipeline(Properties props) {
		super(props);
	}

	@Override
	protected AnnotatorImplementations getAnnotatorImplementations() {
//		String annotatorImplementationsClass = getProperties().getProperty(Constants.CUSTOM_ANNOTATOR_IMPLEMENTATION_CLASS);
//		if (annotatorImplementationsClass != null) {
//			try {
//				return (IMMSAnnotatorImplementations) Class.forName(annotatorImplementationsClass).newInstance();
//			} catch (IllegalAccessException | InstantiationException | ClassNotFoundException e) {
//				throw new RuntimeException(e);
//			}
//		} else {
			return new IMMSAnnotatorImplementations();
//		}
	}

	@Override
	protected synchronized AnnotatorPool getDefaultAnnotatorPool(Properties inputProps,
			AnnotatorImplementations annotatorImplementation) {
		AnnotatorPool superPool = super.getDefaultAnnotatorPool(inputProps, annotatorImplementation);
		superPool.register(Constants.IMMS_SRNDWORDS, IMMSAnnotatorFactories.sorroundingWords(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(Constants.IMMS_POS, IMMSAnnotatorFactories.sorroundingPOS(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(Constants.IMMS_LCOLLOCATION, IMMSAnnotatorFactories.localCollocation(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));
		superPool.register(Constants.IMMS_WORDEMBED, IMMSAnnotatorFactories.wordEmbeddings(inputProps,
				(IMMSAnnotatorImplementations) annotatorImplementation));

		return superPool;
	}


	ICorpusReader getCorpusReader(){
		String className = this.getProperties().getProperty(Constants.IMMS_CORPUS_READER_CLASS,SensEvalLexicalSampleCorpus.class.getName());
		try {
			return (ICorpusReader) Class.forName(className).newInstance();
		} catch (InstantiationException | IllegalAccessException | ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}
	
	Classifier getClassifier(){
		String className = this.getProperties().getProperty(Constants.IMMS_CLASSIFIER_CLASS,LibLinearClassifier.class.getName());		
		try {
			return (Classifier) Class.forName(className).getConstructor(Properties.class).newInstance(this.getProperties());
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
				| NoSuchMethodException | SecurityException | ClassNotFoundException e) {			
			throw new RuntimeException(e);
		}
		
	}
	
}
