package it.uniroma1.lcl.imms.corpus;

import java.util.Iterator;

import edu.stanford.nlp.pipeline.Annotation;

public interface ICorpus {

	Iterator<Annotation> iterator();
}
