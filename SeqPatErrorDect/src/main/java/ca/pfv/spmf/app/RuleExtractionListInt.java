package ca.pfv.spmf.app;

import ca.pfv.spmf.algorithms.sequential_rules.rulegrowth.AlgoERMiner;
import ca.pfv.spmf.input.sequence_database_list_integers.Sequence;
import ca.pfv.spmf.input.sequence_database_list_integers.SequenceDatabase;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by felix on 11.03.17.
 */
public class RuleExtractionListInt {
	
	public static Sequence charStringToSequence(int id, String s) {
		Sequence seq = new Sequence(id);
		for (int i = 0; i < s.length(); i++){
			Character c = s.charAt(i);

			List<Integer> itemset = new ArrayList<>();
			itemset.add((int)c.charValue());
			seq.addItemset(itemset);
		}
		return seq;
	}

	public static Sequence wordStringToSequence(int id, String s, String delimiter) {
		String [] words = s.split(delimiter);
		Sequence seq = new Sequence(id);
		
		for (String word : words) {
			List<Integer> itemset = new ArrayList<>();
			for (int i = 0; i < word.length(); i++) {
				Character c = s.charAt(i);
				itemset.add((int) c.charValue());
			}
			seq.addItemset(itemset);
		}
		return seq;
	}
	
	public static SequenceDatabase readCSV() throws IOException {
		String csvFile = "/home/felix/BlackOak/List_A/inputDB.csv";

		SequenceDatabase db = new SequenceDatabase();

		int i = 0;

		Reader in = new FileReader(csvFile);
		Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(in);
		
		
		for (CSVRecord record : records) {
			
			
			//String ssn = record.get("SSN(String)");
			//Sequence s = charStringToSequence(i, ssn);

			//String zip = record.get("ZIP(String)");
			//Sequence s = charStringToSequence(i, zip);

			String dop = record.get("DOB(String)");
			Sequence s = charStringToSequence(i, dop);

			//String address = record.get("Address(String)");
			//Sequence s = wordStringToSequence(i, address, " ");
			//Sequence s = charStringToSequence(i, address);
			
			
			db.addSequence(s);

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

		AlgoERMiner ruleMiner = new AlgoERMiner(db);
		
		ruleMiner.runAlgorithm(0.1, 0.1, "/home/felix/SequentialPatternErrorDetection/Seq/test.csv");
	}
}
