package it.uniroma1.lcl.imms.corpus.impl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Stack;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.ContextsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.HeadWordStringAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordPositionAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.WordLemmaTag;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.HeadWordLabelAnnotation;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.corpus.ICorpus;
import it.uniroma1.lcl.imms.feature.Feature;

public class SemevalLexicalCorpus implements ICorpus {

	XMLStreamReader xmlReader;
	private Map<String, String> senses = new HashMap<String,String>();
	
	public SemevalLexicalCorpus(String corpusFile) throws FileNotFoundException, XMLStreamException {
		XMLInputFactory f = XMLInputFactory.newInstance();
		xmlReader = f.createXMLStreamReader(new FileInputStream(corpusFile));
	}

	public SemevalLexicalCorpus(String corpusFile,String sensesFile) throws XMLStreamException, IOException, ParseException {
		this(corpusFile);		
		BufferedReader reader=null;
		try{
			reader = new BufferedReader(new FileReader(sensesFile));
			String line;
			int offset=0;
			while((line = reader.readLine())!=null){			
				String[] entry = line.split("\\s+");
				if(entry.length!=3){
					throw new ParseException("Sense file must have three fields per line",offset);
				}
				offset+=line.length();
				senses.put(entry[1],entry[2]);
			}
		} finally {
			if(reader!=null){
				reader.close();
			}
		}
		
	}
	
	@Override
	public Iterator<Annotation> iterator() {
		return new SemevalTextIterator(xmlReader);
	}
	
	class SemevalTextIterator implements Iterator<Annotation> {

		private static final String ELEMENT_CORPUS = "corpus";
		private static final String ELEMENT_LEXELT = "lexelt";
		private static final String ELEMENT_INSTANCE = "instance";
		private static final String ELEMENT_ANSWER = "answer";
		private static final String ELEMENT_CONTEXT = "context";
		private static final String ELEMENT_HEAD = "head";


		private Stack<String> elementsStack;
		private String item;
		private String pos;
		private String lang;
		
		private SemEvalLexicalInstance instance;
		
		public SemevalTextIterator(XMLStreamReader xmlReader) {
			this.elementsStack = new Stack<String>();
		}

		@Override
		public boolean hasNext() {
			try{
				if (instance==null) {
					doNext();
				}
			} catch(Exception e){		
				instance=null;
				throw new RuntimeException(e);
			}
			return instance!=null;
		}

		@Override
		public Annotation next() {
			SemEvalLexicalInstance nextInstance = null;
			if(hasNext()){
				try{
					doNext();				
				} catch(Exception e){		
					instance=null;
					throw new RuntimeException(e);
				}
				if(instance!=null){
					nextInstance = instance;
					instance=null;
				}
			}		
			return nextInstance.toAnnotation();
		}

		private void doNext() throws XMLStreamException {
			while (xmlReader.hasNext()) {
				xmlReader.next();
				if (xmlReader.isStartElement()) {
					String elementName = xmlReader.getLocalName();					
					elementsStack.push(elementName);
					switch (elementName) {
					case ELEMENT_CORPUS:
						check(1, null);
						lang = xmlReader.getAttributeValue(null, "lang");
						break;
					case ELEMENT_LEXELT:
						check(2, ELEMENT_CORPUS);
						item = xmlReader.getAttributeValue(null, "item");
						pos = xmlReader.getAttributeValue(null, "pos");
						break;
					case ELEMENT_INSTANCE:
						check(3, ELEMENT_LEXELT);
						String id = xmlReader.getAttributeValue(null, "id");
						String docsrc = xmlReader.getAttributeValue(null, "docsrc");
						String answer = xmlReader.getAttributeValue(null, "answer");
						if(answer==null){
							answer = senses.get(id);
						}
//						if(answer==null){
//							throw new RuntimeException("Parser error: cannot associate a sense to instance "+id);
//						}
						instance = new SemEvalLexicalInstance(answer,lang, id, item, pos, docsrc);
						return;
					case ELEMENT_ANSWER:
						check(4, ELEMENT_INSTANCE);
						instance.headSense = xmlReader.getAttributeValue(null, "senseid");						
						break;
					case ELEMENT_CONTEXT:
						check(4, ELEMENT_INSTANCE);
						if(instance.isContextSet()){
							throw new RuntimeException("Parser error: multiple contexts for the same instance.");
						}
						break;
					case ELEMENT_HEAD:
						check(5, ELEMENT_CONTEXT);
						if(instance.isHeadSet()){
							throw new RuntimeException("Parser error: multiple target words for the same instance.");
						}
						break;
					}
				} else if (xmlReader.isEndElement()) {
					String closedElementName = xmlReader.getLocalName();
					String elementName = elementsStack.isEmpty() ? null : elementsStack.pop();
					if (!closedElementName.equals(elementName)) {
						throw new RuntimeException(
								"Parser error: missing start tag for element '"
										+ elementName + "': " + elementsStack.toString() + " "+ instance);
					}				
					
					switch (closedElementName) {
					case ELEMENT_INSTANCE:
						return;
					case ELEMENT_HEAD:
						if (instance.getHead().isEmpty()) {
							throw new RuntimeException(
									"Parser error: target word cannot be null");
						}
						break;
					}

				} else if (xmlReader.isCharacters()){
					String elementName = elementsStack.peek();
					if(ELEMENT_CONTEXT.equals(elementName)) {
						if (instance.getHead().isEmpty()) {
							instance.appendLeftContext(xmlReader.getText());
						} else {
							instance.appendRightContext(xmlReader.getText());
						}
					} else if (ELEMENT_HEAD.equals(elementName)) {
						instance.appendHead(xmlReader.getText());
					}
				}
			}
		}

		private void check(int elementDepth, String parentElement) {
			if (elementsStack.size() != elementDepth
					|| (elementDepth > 1 && !elementsStack.elementAt(
							elementDepth - 2).equals(parentElement))) {
				throw new RuntimeException("Parser error: unexpected element '"
						+ elementsStack.peek() + "' at " + elementsStack.toString());
			}

		}

		
		@Override
		public void remove() {
			// TODO Auto-generated method stub

		}

	}

	class SemEvalLexicalInstance{

		private String lang;
		private String id;
		private String item;
		private String pos;
		private String docsrc;
		private StringBuffer leftContext;
		private StringBuffer head;
		private StringBuffer rightContext;
		
		private boolean contextSet;
		private boolean headSet;
		private String headSense;

		public SemEvalLexicalInstance(String headSense, String lang, String id, String item, String pos,
				String docsrc) {
			this.lang = lang;
			this.id = id;
			this.item = item;
			this.pos = pos;
			this.headSense = headSense;
			this.docsrc = docsrc;
			this.leftContext = new StringBuffer();
			this.rightContext = new StringBuffer();
			this.head = new StringBuffer();
		}

		public String getLeftContext() {
			return leftContext.toString();
		}

		public String getRightContext() {
			return rightContext.toString();
		}

		public String getHead() {
			return head.toString();
		}	

		public void appendLeftContext(String text) {
			contextSet = true;
			leftContext.append(text);
		}

		public void appendRightContext(String text) {
			contextSet = true;
			rightContext.append(text);
		}

		public void appendHead(String text) {
			headSet = true;
			head.append(text);
		}

		public boolean isContextSet() {
			return contextSet;
		}

		public boolean isHeadSet() {
			return headSet;
		}
		
		@Override
		public String toString() {
			return this.getLeftContext()+"_HEAD_START_"+this.getHead()+"_HEAD_END_"+this.getRightContext();
		}
		
		Annotation toAnnotation(){
			Annotation anno = new Annotation(this.getLeftContext()+this.getHead()+this.getRightContext());
			anno.set(CoreAnnotations.DocIDAnnotation.class, this.id);
			anno.set(Constants.LexicalElementAnnotation.class,this.item);
			
			CoreLabel head = new CoreLabel();
			head.setBeginPosition(this.getLeftContext().length());
			head.setEndPosition(this.getLeftContext().length()+this.getHead().length());			
			head.set(WordSenseAnnotation.class, headSense==null ? Constants.UNKNOWN_SENSE : headSense);
			head.set(Constants.FeaturesAnnotation.class, new ArrayList<Feature>());
			anno.set(Constants.HeadAnnotation.class, head);
			
			HashMap<String, String> hm = new HashMap<String,String>();
			hm.put("lang", this.lang);
			hm.put("id", this.id);
			hm.put("item", this.item);
			hm.put("pos", this.pos);
			hm.put("docsrc", this.docsrc);
			hm.put("leftContext", this.getLeftContext());
			hm.put("rightContext", this.getRightContext());		
			hm.put("head", this.getHead());
			
			return anno;
		}

	}
}
