package ca.pfv.spmf.app;

import ca.pfv.spmf.algorithms.sequential_rules.topseqrules_and_tns.AlgoTNS;
import ca.pfv.spmf.algorithms.sequential_rules.topseqrules_and_tns.Rule;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.AlgoCM_ClaSP;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.dataStructures.creators.AbstractionCreator;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.dataStructures.creators.AbstractionCreator_Qualitative;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.idlists.creators.IdListCreator;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.idlists.creators.IdListCreatorStandard_Map;
import ca.pfv.spmf.datastructures.redblacktree.RedBlackTree;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.dataStructures.Item;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.dataStructures.Itemset;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.dataStructures.Sequence;
import ca.pfv.spmf.algorithms.sequentialpatterns.clasp_AGP.dataStructures.database.SequenceDatabase;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by felix on 11.03.17.
 */
public class RuleExtractionClosedSeqPat {
	
	public static Sequence charStringToSequence(int id, String s, SequenceDatabase db, List<Integer> sizeItemsetsList) {
		Sequence seq = new Sequence(id);
		for (int i = 0; i < s.length(); i++){
			Character c = s.charAt(i);

			Itemset itemset = new Itemset();
			int item = (int) c.charValue();
			itemset.addItem(new Item(item));
			seq.addItemset(itemset);
			sizeItemsetsList.add(seq.length());
		}
		return seq;
	}

	public static Sequence wordStringToSequence(int id, String s, String delimiter, SequenceDatabase db, List<Integer> sizeItemsetsList) {
		String [] words = s.split(delimiter);
		Sequence seq = new Sequence(id);
		
		for (String word : words) {
			Itemset itemset = new Itemset();
			for (int i = 0; i < word.length(); i++) {
				Character c = s.charAt(i);
				Integer item = (int) c.charValue();
				itemset.addItem(new Item(item));
			}
			seq.addItemset(itemset);
			sizeItemsetsList.add(seq.length());
		}
		return seq;
	}
	
	public static SequenceDatabase readCSV() throws IOException {
		//String csvFile = "/home/felix/BlackOak/List_A/inputDB.csv";
		String csvFile = "/home/felix/BlackOak/List_A/groundDB.csv";

		AbstractionCreator abstractionCreator = AbstractionCreator_Qualitative.getInstance();
		IdListCreator idListCreator = IdListCreatorStandard_Map.getInstance();

		SequenceDatabase db = new SequenceDatabase(abstractionCreator, idListCreator);

		int i = 0;

		Reader in = new FileReader(csvFile);
		Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(in);


		
		
		for (CSVRecord record : records) {
			//String ssn = record.get(10); //SSN
			//Sequence s = charStringToSequence(i, ssn, db);

			String state = record.get(6); //State
			List<Integer> sizeItemsetsList = new ArrayList<Integer>();
			Sequence s = charStringToSequence(i, state, db, sizeItemsetsList);

			//String zip = record.get(7); //ZIP
			//Sequence s = charStringToSequence(i, zip, db);
			
			//String dop = record.get(11); //DOB
			//Sequence s = charStringToSequence(i, dop, db);

			//String address = record.get(4);
			//Sequence s = wordStringToSequence(i, address, " ", db);
			//Sequence s = charStringToSequence(i, address, db);
			
			
			db.addSequence(s);
			db.nSequences++;
			db.sequencesLengths.put(s.getId(), s.length());
			db.sequenceItemsetSize.put(s.getId(), sizeItemsetsList);

			i++;
			
			if (record.size() < 12) {
				System.err.println(record.toString());
			}
		}
		return db;
	}

	public static void main(String [] args) throws IOException {
		SequenceDatabase db = readCSV();
		
		System.out.println("space: " + Character.getNumericValue(' '));

		//AlgoTopSeqRules ruleMiner = new AlgoTopSeqRules();		
		AlgoCM_ClaSP patternMiner = new AlgoCM_ClaSP(0.1, db.abstractionCreator, true, true);
		
		patternMiner.runAlgorithm(db, true, true, "test.txt", false);
		patternMiner.printStatistics();
		
	}
}
