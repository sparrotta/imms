package it.uniroma1.lcl.imms.corpus;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;

import edu.stanford.nlp.pipeline.Annotation;

public interface ICorpusReader {

	void loadCorpus(String corpusFile) throws FileNotFoundException, IOException;
	void loadAnswers(String answersFile) throws FileNotFoundException, IOException;
	Iterator<Annotation> iterator();
}
