package it.uniroma1.lcl.imms.task.impl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;
import java.util.Stack;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;

import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.CoreAnnotations.IDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.DocSourceAnnotation;
import it.uniroma1.lcl.imms.Constants.LexicalItemAnnotation;
import it.uniroma1.lcl.imms.annotator.feature.Feature;
import it.uniroma1.lcl.imms.task.ITaskHandler;
import net.didion.jwnl.JWNL;
import net.didion.jwnl.JWNLException;
import net.didion.jwnl.data.IndexWord;
import net.didion.jwnl.dictionary.Dictionary;

public class SensEval2LexicalSampleTask implements ITaskHandler {

	XMLStreamReader xmlReader;
	private Map<String, String> answers = new HashMap<String,String>();
	Dictionary dictionary;
	String resultDir;
	public SensEval2LexicalSampleTask(Properties properties) {
		resultDir = properties.getProperty(Constants.PROPERTY_TASK_RESULT_DIR);
		if(!JWNL.isInitialized()){
			try {
				JWNL.initialize(new FileInputStream(properties.getProperty(Constants.PROPERTY_JWNL_PROP_FILE)));
			} catch (FileNotFoundException|JWNLException e) {
				e.printStackTrace();
			}		
		}
		dictionary = Dictionary.getInstance();
	}
	
	@Override
	public Iterator<Annotation> iterator() {
		return new SensEval2LexicalSampleCorpusIterator(xmlReader);
	}
	
	public void writeResults(String lexElem, RVFDataset<String,String> dataset){
		String resultFilename = resultDir+Constants.fileSeparator+lexElem+".result";
		OutputStreamWriter os =null;
		try{
			os = new OutputStreamWriter(new FileOutputStream(resultFilename));
			String[] lemmaPosArr = lexElem.split("\\.");
			IndexWord word=null;			
			try {
				word = dictionary.lookupIndexWord(Constants.posTagMap.get(lemmaPosArr[1]), lemmaPosArr[0]);
			} catch (JWNLException e) {}
						
			for(int i=0; i<dataset.size();i++){
				String answer = dataset.getRVFDatum(i).label();
				if(answer==null && word!=null){
					try {
						answer = word.getSense(1).getSenseKey(lemmaPosArr[0]);
					} catch (JWNLException e) {
						
					}
				}
				if(answer==null){
					answer="U";
				}
				os.append(lemmaPosArr[0]+" "+dataset.getRVFDatumId(i)+" "+answer+"\n");
			}		
		} catch(IOException e){
			throw new RuntimeException(e);
		} finally {
			if(os!=null){
				try {
					os.flush();
					os.close();
				} catch (IOException e) {					
					e.printStackTrace();
				}
			}
		}
	}
	
	

	
	class SensEval2LexicalSampleCorpusIterator implements Iterator<Annotation> {

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
		
		private SenseEvalLexicalItemInstance instance;
		
		public SensEval2LexicalSampleCorpusIterator(XMLStreamReader xmlReader) {
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
			SenseEvalLexicalItemInstance nextInstance = null;
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
							answer = answers.get(id);
						}
//						if(answer==null){
//							throw new RuntimeException("Parser error: cannot associate a sense to instance "+id);
//						}
						instance = new SenseEvalLexicalItemInstance(answer,lang, id, item, pos, docsrc);
						return;
					case ELEMENT_ANSWER:
						check(4, ELEMENT_INSTANCE);
						if(instance.headSense==null){
							instance.headSense = xmlReader.getAttributeValue(null, "senseid");
						}												
						break;
					case ELEMENT_CONTEXT:
						check(4, ELEMENT_INSTANCE);
						if(instance.isContextSet()){
							throw new RuntimeException("Parser error: multiple contexts for the same instance.");
						}
						break;
					case ELEMENT_HEAD:
						check(5, ELEMENT_CONTEXT);
						if(instance.isHeadLocked()){
							throw new RuntimeException("Parser error: multiple target words for the same instance: "+instance.toString());
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
									"Parser error: head word cannot be null");
						} else {
							instance.lockHead();
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
						if(!instance.isHeadLocked()){
							instance.appendHead(xmlReader.getText());
						} else {
							instance.appendRightContext(xmlReader.getText());
						}
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

	class SenseEvalLexicalItemInstance{

		private String lang;
		private String pos;
		private String id;
		private String item;		
		private String docsrc;
		private StringBuffer leftContext;
		private StringBuffer head;
		private StringBuffer rightContext;
		
		private boolean contextSet;
		private boolean headLock;
		private String headSense;

		public SenseEvalLexicalItemInstance(String headSense, String lang, String id, String item, String pos, String docsrc) {
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

		public void lockHead() {
			headLock = true;			
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
			head.append(text);
		}

		public boolean isContextSet() {
			return contextSet;
		}

		public boolean isHeadLocked() {
			return headLock;
		}
		
		@Override
		public String toString() {
			return this.getLeftContext()+"_HEAD_START_"+this.getHead()+"_HEAD_END_"+this.getRightContext();
		}
		
		Annotation toAnnotation(){
			Annotation anno = new Annotation(this.getLeftContext()+" "+this.getHead()+" "+this.getRightContext());			
			
			CoreLabel head = new CoreLabel();
			head.set(IDAnnotation.class,this.id);
			head.setBeginPosition(this.getLeftContext().length()+1);
			head.setEndPosition(this.getLeftContext().length()+1+this.getHead().length());			
			head.set(DocSourceAnnotation.class, docsrc);
			head.set(LexicalItemAnnotation.class,this.item);
			head.set(WordSenseAnnotation.class, headSense==null ? Constants.UNKNOWN_SENSE : headSense);
			head.set(Constants.FeaturesAnnotation.class, new ArrayList<Feature>());
			
			anno.set(Constants.HeadsAnnotation.class, Arrays.asList(new CoreLabel[]{head}));
			
			return anno;
		}

	}

	@Override
	public void loadCorpus(String corpusFile) throws IOException {
		XMLInputFactory f = XMLInputFactory.newInstance();
//		f.setProperty(XMLInputFactory.IS_REPLACING_ENTITY_REFERENCES, false);
		try {
			xmlReader = f.createXMLStreamReader(new FileInputStream(corpusFile));
		} catch (XMLStreamException e) {
			throw new IOException(e);
		}		
	}

	@Override
	public void loadAnswers(String answersFile) throws FileNotFoundException, IOException {
		BufferedReader reader=null;
		try{
			reader = new BufferedReader(new FileReader(answersFile));
			String line;
			int offset=0;
			while((line = reader.readLine())!=null){			
				String[] entry = line.split("\\s+");
				if(entry.length<3){
					throw new IOException(new ParseException("Answers file must have at least three fields per line",offset));
				}
				offset+=line.length();
				
				answers.put(entry[1],entry[2]);
			}
		} finally {
			if(reader!=null){
				reader.close();
			}
		}
		
	}
}
