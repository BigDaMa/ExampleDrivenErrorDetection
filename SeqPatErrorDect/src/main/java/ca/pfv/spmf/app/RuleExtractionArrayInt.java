package ca.pfv.spmf.app;

import ca.pfv.spmf.algorithms.sequential_rules.rulegrowth.AlgoERMiner;
import ca.pfv.spmf.algorithms.sequential_rules.topseqrules_and_tns.AlgoTNS;
import ca.pfv.spmf.algorithms.sequential_rules.topseqrules_and_tns.AlgoTopSeqRules;
import ca.pfv.spmf.algorithms.sequential_rules.topseqrules_and_tns.Rule;
import ca.pfv.spmf.datastructures.redblacktree.RedBlackTree;
import ca.pfv.spmf.input.sequence_database_array_integers.Sequence;
import ca.pfv.spmf.input.sequence_database_array_integers.SequenceDatabase;
import org.apache.commons.cli.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by felix on 11.03.17.
 */
public class RuleExtractionArrayInt {
	
	public static Sequence charStringToSequence(int id, String s, SequenceDatabase db) {
		Sequence seq = new Sequence();
		for (int i = 0; i < s.length(); i++){
			Character c = s.charAt(i);

			List<Integer> itemset = new ArrayList<>();
			int item = (int) c.charValue();
			itemset.add(item);
			seq.addItemset(itemset.toArray());

			if(item >= db.maxItem){
				db.maxItem = item;
			}
			// we update the minimum item for statistics
			if(item < db.minItem){
				db.minItem = item;
			}
		}
		return seq;
	}

	public static Sequence wordStringToSequence(int id, String s, String delimiter, SequenceDatabase db) {
		String [] words = s.split(delimiter);
		Sequence seq = new Sequence();
		
		for (String word : words) {
			List<Integer> itemset = new ArrayList<>();
			for (int i = 0; i < word.length(); i++) {
				Character c = s.charAt(i);
				int item = (int) c.charValue();
				itemset.add(item);

				if(item >= db.maxItem){
					db.maxItem = item;
				}
				// we update the minimum item for statistics
				if(item < db.minItem){
					db.minItem = item;
				}
			}
			seq.addItemset(itemset.toArray());
		}
		return seq;
	}
	
	public static SequenceDatabase readCSV(String csvFile, int columnID) throws IOException {
		SequenceDatabase db = new SequenceDatabase();

		int i = 0;

		Reader in = new FileReader(csvFile);
		Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(in);
		
		
		for (CSVRecord record : records) {
			//String ssn = record.get(10); //SSN
			//Sequence s = charStringToSequence(i, ssn, db);

			//String state = record.get(6); //State
			//Sequence s = charStringToSequence(i, state, db);

			//String zip = record.get(7); //ZIP
			//Sequence s = charStringToSequence(i, zip, db);
			
			//String dop = record.get(11); //DOB
			//Sequence s = charStringToSequence(i, dop, db);

			//String address = record.get(4);
			//Sequence s = wordStringToSequence(i, address, " ", db);
			//Sequence s = charStringToSequence(i, address, db);

			String column = record.get(columnID); //State
			Sequence s = charStringToSequence(i, column, db);
			
			db.addSequence(s);

			i++;
			
			if (record.size() < 12) {
				System.err.println(record.toString());
			}
		}
		return db;
	}

	public static void main(String [] args) throws IOException {
		
		System.out.println("hallo");

		Options options = new Options();
		Option input = OptionBuilder.withArgName( "input" )
			.hasArg(true)
			.withDescription(  "input csv file" )
			.create( "input");
		options.addOption(input);

		Option output = OptionBuilder.withArgName( "output" )
			.hasArg(true)
			.withDescription(  "output rule csv file" )
			.create( "output");
		options.addOption(output);

		Option columnIdArg = OptionBuilder.withArgName( "cindex" )
			.hasArg(true)
			.withDescription(  "index of the column of which we create the rules (0 .. n-1)\\ndefault:first(0) " +
				"column" )
			.create( "cindex");
		options.addOption(columnIdArg);

		Option numberRulesArg = OptionBuilder.withArgName( "nrules" )
			.hasArg(true)
			.withDescription(  "number of rules we generate maximally\\ndefault:100 " +
				"rules" )
			.create( "nrules");
		options.addOption(numberRulesArg);

		CommandLineParser parser = new DefaultParser();
		CommandLine cmd = null;

		String csvFile = "";
		String outputFile = "";
		int columnID = 0;
		int numberRules = 0;

		try {
			cmd = parser.parse(options, args);

			csvFile = cmd.getOptionValue("input", "/home/felix/BlackOak/List_A/inputDB.csv");
			System.out.println("here: " + csvFile);

			outputFile = cmd.getOptionValue("output", "/tmp/rules.csv");
			System.out.println("here: " + outputFile);

			columnID = Integer.parseInt(cmd.getOptionValue("cindex", "0"));
			System.out.println("here: " + columnID);
		
			numberRules = Integer.parseInt(cmd.getOptionValue("nrules", "100"));
			System.out.println("here: " + numberRules);

		} catch (Exception e) {
			e.printStackTrace();

			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "GenerateTopKRules", options );
			return;
		}
		
		
		//csvFile = "/home/felix/BlackOak/List_A/inputDB.csv";
		//csvFile = "/home/felix/BlackOak/List_A/groundDB.csv";
		
		SequenceDatabase db = readCSV(csvFile, columnID);
		
		AlgoTNS ruleMiner = new AlgoTNS();
		RedBlackTree<Rule> rules = ruleMiner.runAlgorithm(numberRules, db, 0.01, 1);
		
		/*
		for (Rule r : rules) {
			System.out.println(r.toStringChar());
		}*/

		try{
			PrintWriter writer = new PrintWriter(outputFile, "UTF-8");
			writer.println("Rule;Absolute_Support;Confidence");
			for (Rule r : rules) {
				writer.println(r.toStringChar());
			}
			writer.close();
		} catch (IOException e) {
			// do something
		}
	}
}
