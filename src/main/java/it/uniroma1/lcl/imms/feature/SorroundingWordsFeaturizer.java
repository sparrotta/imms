package it.uniroma1.lcl.imms.feature;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.HeadAnnotation;

import java.util.*;

public class SorroundingWordsFeaturizer implements Annotator {


	public static final String DEFAULT_WINDOWSIZE = "2";
	
	Integer windowSize;
	
	
	List<String> stopWords = new ArrayList<String>();
	
	
	public SorroundingWordsFeaturizer(Properties properties) {
		this.stopWords.add("a");
		this.stopWords.add("about");
		this.stopWords.add("above");
		this.stopWords.add("across");
		this.stopWords.add("after");
		this.stopWords.add("afterwards");
		this.stopWords.add("again");
		this.stopWords.add("against");
		this.stopWords.add("albeit");
		this.stopWords.add("all");
		this.stopWords.add("almost");
		this.stopWords.add("alone");
		this.stopWords.add("along");
		this.stopWords.add("already");
		this.stopWords.add("also");
		this.stopWords.add("although");
		this.stopWords.add("always");
		this.stopWords.add("among");
		this.stopWords.add("amongst");
		this.stopWords.add("an");
		this.stopWords.add("and");
		this.stopWords.add("another");
		this.stopWords.add("any");
		this.stopWords.add("anyhow");
		this.stopWords.add("anyone");
		this.stopWords.add("anything");
		this.stopWords.add("anywhere");
		this.stopWords.add("are");
		this.stopWords.add("around");
		this.stopWords.add("as");
		this.stopWords.add("at");
		this.stopWords.add("b");
		this.stopWords.add("be");
		this.stopWords.add("became");
		this.stopWords.add("because");
		this.stopWords.add("become");
		this.stopWords.add("becomes");
		this.stopWords.add("becoming");
		this.stopWords.add("been");
		this.stopWords.add("before");
		this.stopWords.add("beforehand");
		this.stopWords.add("behind");
		this.stopWords.add("being");
		this.stopWords.add("below");
		this.stopWords.add("beside");
		this.stopWords.add("besides");
		this.stopWords.add("between");
		this.stopWords.add("beyond");
		this.stopWords.add("both");
		this.stopWords.add("but");
		this.stopWords.add("by");
		this.stopWords.add("c");
		this.stopWords.add("can");
		this.stopWords.add("cannot");
		this.stopWords.add("co");
		this.stopWords.add("could");
		this.stopWords.add("d");
		this.stopWords.add("down");
		this.stopWords.add("during");
		this.stopWords.add("e");
		this.stopWords.add("each");
		this.stopWords.add("eg");
		this.stopWords.add("either");
		this.stopWords.add("else");
		this.stopWords.add("elsewhere");
		this.stopWords.add("enough");
		this.stopWords.add("etc");
		this.stopWords.add("even");
		this.stopWords.add("ever");
		this.stopWords.add("every");
		this.stopWords.add("everyone");
		this.stopWords.add("everything");
		this.stopWords.add("everywhere");
		this.stopWords.add("except");
		this.stopWords.add("f");
		this.stopWords.add("few");
		this.stopWords.add("for");
		this.stopWords.add("former");
		this.stopWords.add("formerly");
		this.stopWords.add("from");
		this.stopWords.add("further");
		this.stopWords.add("g");
		this.stopWords.add("h");
		this.stopWords.add("had");
		this.stopWords.add("has");
		this.stopWords.add("have");
		this.stopWords.add("he");
		this.stopWords.add("hence");
		this.stopWords.add("her");
		this.stopWords.add("here");
		this.stopWords.add("hereafter");
		this.stopWords.add("hereby");
		this.stopWords.add("herein");
		this.stopWords.add("hereupon");
		this.stopWords.add("hers");
		this.stopWords.add("herself");
		this.stopWords.add("him");
		this.stopWords.add("himself");
		this.stopWords.add("his");
		this.stopWords.add("how");
		this.stopWords.add("however");
		this.stopWords.add("i");
		this.stopWords.add("ie");
		this.stopWords.add("if");
		this.stopWords.add("in");
		this.stopWords.add("inc");
		this.stopWords.add("indeed");
		this.stopWords.add("into");
		this.stopWords.add("is");
		this.stopWords.add("it");
		this.stopWords.add("its");
		this.stopWords.add("itself");
		this.stopWords.add("j");
		this.stopWords.add("k");
		this.stopWords.add("l");
		this.stopWords.add("latter");
		this.stopWords.add("latterly");
		this.stopWords.add("least");
		this.stopWords.add("less");
		this.stopWords.add("ltd");
		this.stopWords.add("m");
		this.stopWords.add("many");
		this.stopWords.add("may");
		this.stopWords.add("me");
		this.stopWords.add("meanwhile");
		this.stopWords.add("might");
		this.stopWords.add("more");
		this.stopWords.add("moreover");
		this.stopWords.add("most");
		this.stopWords.add("mostly");
		this.stopWords.add("much");
		this.stopWords.add("must");
		this.stopWords.add("my");
		this.stopWords.add("myself");
		this.stopWords.add("n");
		this.stopWords.add("namely");
		this.stopWords.add("neither");
		this.stopWords.add("never");
		this.stopWords.add("nevertheless");
		this.stopWords.add("next");
		this.stopWords.add("no");
		this.stopWords.add("nobody");
		this.stopWords.add("none");
		this.stopWords.add("noone");
		this.stopWords.add("nor");
		this.stopWords.add("not");
		this.stopWords.add("nothing");
		this.stopWords.add("now");
		this.stopWords.add("nowhere");
		this.stopWords.add("o");
		this.stopWords.add("of");
		this.stopWords.add("off");
		this.stopWords.add("often");
		this.stopWords.add("on");
		this.stopWords.add("once");
		this.stopWords.add("one");
		this.stopWords.add("only");
		this.stopWords.add("onto");
		this.stopWords.add("or");
		this.stopWords.add("other");
		this.stopWords.add("others");
		this.stopWords.add("otherwise");
		this.stopWords.add("our");
		this.stopWords.add("ours");
		this.stopWords.add("ourselves");
		this.stopWords.add("out");
		this.stopWords.add("over");
		this.stopWords.add("own");
		this.stopWords.add("p");
		this.stopWords.add("per");
		this.stopWords.add("perhaps");
		this.stopWords.add("q");
		this.stopWords.add("r");
		this.stopWords.add("rather");
		this.stopWords.add("s");
		this.stopWords.add("same");
		this.stopWords.add("seem");
		this.stopWords.add("seemed");
		this.stopWords.add("seeming");
		this.stopWords.add("seems");
		this.stopWords.add("several");
		this.stopWords.add("she");
		this.stopWords.add("should");
		this.stopWords.add("since");
		this.stopWords.add("so");
		this.stopWords.add("some");
		this.stopWords.add("somehow");
		this.stopWords.add("someone");
		this.stopWords.add("something");
		this.stopWords.add("sometime");
		this.stopWords.add("sometimes");
		this.stopWords.add("somewhere");
		this.stopWords.add("still");
		this.stopWords.add("such");
		this.stopWords.add("t");
		this.stopWords.add("than");
		this.stopWords.add("that");
		this.stopWords.add("the");
		this.stopWords.add("their");
		this.stopWords.add("them");
		this.stopWords.add("themselves");
		this.stopWords.add("then");
		this.stopWords.add("thence");
		this.stopWords.add("there");
		this.stopWords.add("thereafter");
		this.stopWords.add("thereby");
		this.stopWords.add("therefore");
		this.stopWords.add("therein");
		this.stopWords.add("thereupon");
		this.stopWords.add("these");
		this.stopWords.add("they");
		this.stopWords.add("this");
		this.stopWords.add("those");
		this.stopWords.add("though");
		this.stopWords.add("through");
		this.stopWords.add("throughout");
		this.stopWords.add("thru");
		this.stopWords.add("thus");
		this.stopWords.add("to");
		this.stopWords.add("together");
		this.stopWords.add("too");
		this.stopWords.add("toward");
		this.stopWords.add("towards");
		this.stopWords.add("u");
		this.stopWords.add("under");
		this.stopWords.add("until");
		this.stopWords.add("up");
		this.stopWords.add("upon");
		this.stopWords.add("v");
		this.stopWords.add("very");
		this.stopWords.add("via");
		this.stopWords.add("w");
		this.stopWords.add("was");
		this.stopWords.add("we");
		this.stopWords.add("well");
		this.stopWords.add("were");
		this.stopWords.add("what");
		this.stopWords.add("whatever");
		this.stopWords.add("whatsoever");
		this.stopWords.add("when");
		this.stopWords.add("whence");
		this.stopWords.add("whenever");
		this.stopWords.add("whensoever");
		this.stopWords.add("where");
		this.stopWords.add("whereafter");
		this.stopWords.add("whereas");
		this.stopWords.add("whereat");
		this.stopWords.add("whereby");
		this.stopWords.add("wherefrom");
		this.stopWords.add("wherein");
		this.stopWords.add("whereinto");
		this.stopWords.add("whereof");
		this.stopWords.add("whereon");
		this.stopWords.add("whereto");
		this.stopWords.add("whereunto");
		this.stopWords.add("whereupon");
		this.stopWords.add("wherever");
		this.stopWords.add("wherewith");
		this.stopWords.add("whether");
		this.stopWords.add("which");
		this.stopWords.add("whichever");
		this.stopWords.add("whichsoever");
		this.stopWords.add("while");
		this.stopWords.add("whilst");
		this.stopWords.add("whither");
		this.stopWords.add("who");
		this.stopWords.add("whoever");
		this.stopWords.add("whole");
		this.stopWords.add("whom");
		this.stopWords.add("whomever");
		this.stopWords.add("whomsoever");
		this.stopWords.add("whose");
		this.stopWords.add("whosoever");
		this.stopWords.add("why");
		this.stopWords.add("will");
		this.stopWords.add("with");
		this.stopWords.add("within");
		this.stopWords.add("without");
		this.stopWords.add("would");
		this.stopWords.add("x");
		this.stopWords.add("yet");
		this.stopWords.add("you");
		this.stopWords.add("your");
		this.stopWords.add("yours");
		this.stopWords.add("yourself");
		this.stopWords.add("yourselves");
		this.stopWords.add("z");
		this.stopWords.add("say");
		this.stopWords.add("says");
		this.stopWords.add("said");
		this.stopWords.add("do");
		this.stopWords.add("n't");
		this.stopWords.add("'ve");
		this.stopWords.add("'d");
		this.stopWords.add("'m");
		this.stopWords.add("'s");
		this.stopWords.add("'re");
		this.stopWords.add("'ll");
		this.stopWords.add("-lrb-");
		this.stopWords.add("-rrb-");
		this.stopWords.add("-lsb-");
		this.stopWords.add("-rsb-");
		this.stopWords.add("-lcb-");
		this.stopWords.add("-rcb-");
		for(String word : properties.getProperty(Constants.PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD,"").split("\\s")){
			stopWords.add(word.trim().toLowerCase());
		}
		windowSize = Integer.valueOf(properties.getProperty(Constants.PROPERTY_IMMS_SRNDWORDS_WINDOWSIZE,DEFAULT_WINDOWSIZE));
	}

	@Override
	public void annotate(Annotation annotation) {
			
		CoreLabel head = annotation.get(HeadAnnotation.class);
		List<String> before = new ArrayList<String>();
		List<String> after = new ArrayList<String>();
				
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		for (CoreMap sentence : sentences) {
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				if (token.endPosition() < head.beginPosition() && filter(token)) {
					before.add(token.lemma().toLowerCase());									
				} else if (token.beginPosition() > head.endPosition() && filter(token)) {
					after.add(token.lemma().toLowerCase());					
				}				
			}
		}
		List<Feature> features = new ArrayList<Feature>();
		int beforeSize = before.size();
		for(int i=Math.max(0,beforeSize-windowSize); i<beforeSize; i++){
			features.add(new Feature<Boolean>("S_"+before.get(i),true));
		}
		int afterSize = after.size();
		for(int i=0;i<Math.min(windowSize,afterSize);i++){
			features.add(new Feature<Boolean>("S_"+after.get(i),true));
		}
		head.get(Constants.FeaturesAnnotation.class).addAll(features);		
	}

	boolean filter(CoreLabel token){
		return Constants.PREDICATE_IS_WORD.test(token.lemma()) && !stopWords.contains(token.lemma());
	}
	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.singleton(Constants.IMMS_SRNDWORDS_REQUIREMENT);
	}

	@Override
	public Set<Requirement> requires() {
		return TOKENIZE_SSPLIT_POS_LEMMA;
	}

}
