import csv

from ml.datasets.mohammad import MohammadDataSet
from ml.plot.old.runtime_all import Plotter

data = MohammadDataSet("books", 30, 30, 10)

print data.shape

data.clean_pd.to_csv("clean_books.csv", index=None, escapechar='\\', encoding='utf-8', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)



real_time = [20.079864978790283, 29.251036882400513, 38.62403988838196, 52.069242000579834, 64.27451086044312, 77.2811348438263, 82.229651927948, 87.0179238319397, 92.97003984451294, 99.19389796257019, 104.46019291877747, 109.84354901313782, 117.04936981201172, 125.17161297798157, 130.7246458530426, 136.9955279827118, 143.94308590888977, 152.57093286514282, 158.59668397903442, 165.1645529270172, 173.0561089515686, 181.59059596061707, 188.11917400360107, 194.78248381614685, 204.41021299362183, 212.30275297164917, 220.5078408718109, 229.99510288238525, 237.39066195487976, 244.92486786842346, 255.69548392295837, 263.9607298374176, 275.4024260044098]

fscore_0 = []
fscore_0.append([0.0, 0.0, 0.0, 0.0039787798408488064, 0.0039787798408488064, 0.18946208922342572, 0.23134863701578193, 0.23876901308807927, 0.3056083650190114, 0.40456477959275006, 0.40063593004769477, 0.44421958006577283, 0.44421958006577283, 0.4742166296570442, 0.61049902786778998, 0.61049902786778998, 0.63908338637810302, 0.66600000000000004, 0.66600000000000004, 0.70450859552384049, 0.74180327868852458, 0.74180327868852458, 0.78258000659848237, 0.76982842343800584, 0.76982842343800584, 0.77611940298507454, 0.76638444653496396, 0.76638444653496396, 0.78266331658291466, 0.82407102926668863, 0.82407102926668863, 0.82252222588080337, 0.82512396694214885])
fscore_0.append([0.0, 0.0, 0.0, 0.0039787798408488064, 0.0039787798408488064, 0.1887277400403892, 0.30974815830049685, 0.31061643835616443, 0.32996880161497522, 0.32996880161497522, 0.3331506849315069, 0.36195716250736887, 0.36195716250736887, 0.36768149882903978, 0.48389590311775316, 0.48389590311775316, 0.48809523809523814, 0.4526748971193415, 0.4526748971193415, 0.47496423462088699, 0.56195066628863055, 0.56195066628863055, 0.58790749512398988, 0.66302952503209245, 0.66302952503209245, 0.71384803921568629, 0.62589356632247817, 0.62589356632247817, 0.64802717533315912, 0.74009603841536609, 0.74009603841536609, 0.75170572530406399, 0.77502295684113864])
fscore_0.append([0.0, 0.0, 0.0, 0.0039787798408488064, 0.0039787798408488064, 0.19560272934040943, 0.3111584175980131, 0.32456904057113006, 0.33925925925925932, 0.34813041075704548, 0.3473762010347376, 0.42492527017705217, 0.42492527017705217, 0.43470277714023414, 0.41690263559137442, 0.41690263559137442, 0.43807486631016046, 0.55340071018847314, 0.55340071018847314, 0.56896551724137934, 0.50308202939781888, 0.50308202939781888, 0.5086976962858486, 0.68802588996763747, 0.68802588996763747, 0.69397116644823065, 0.70302627203192558, 0.70302627203192558, 0.72998379254457046, 0.76035302104548541, 0.76035302104548541, 0.76833845104060039, 0.80326704545454541])
fscore_0.append([0.0, 0.0, 0.0, 0.0039787798408488064, 0.0039787798408488064, 0.20070560564484519, 0.32847778587035686, 0.3323305609626615, 0.43796675499879545, 0.43796675499879545, 0.44406531805995614, 0.55885997521685249, 0.55885997521685249, 0.57485029940119758, 0.64733261725742919, 0.64733261725742919, 0.6794158553546592, 0.61678146524733868, 0.61678146524733868, 0.70601919162547233, 0.82156260661890135, 0.82156260661890135, 0.83925549915397624, 0.81139489194499015, 0.81139489194499015, 0.80842659644502946, 0.78859705317104412, 0.78859705317104412, 0.79462571976967378, 0.82941571524513091, 0.82941571524513091, 0.82913729439409201, 0.84549503254539227])
fscore_0.append([0.0, 0.0, 0.0, 0.0039787798408488064, 0.0039787798408488064, 0.20078277886497062, 0.32836363636363636, 0.32927494094130472, 0.32915531335149867, 0.32915531335149867, 0.33188405797101445, 0.40617199909235308, 0.40617199909235308, 0.47328899377815914, 0.64825930372148866, 0.64825930372148866, 0.69280280866003507, 0.5961155378486056, 0.5961155378486056, 0.57974683544303796, 0.58252427184466016, 0.58252427184466016, 0.59615384615384615, 0.74262139003490957, 0.74262139003490957, 0.75993740219092332, 0.80212483399734391, 0.80212483399734391, 0.80240320427236322, 0.77224542242210092, 0.77224542242210092, 0.77458109389819796, 0.77520350657482784])

dboost_fscore = 0.0967
runtime_dboost_sec = 21 + 23

nadeef_fscore = [0.0, 0.40632486823191177, 0.41042654028436021, 0.4322073114298936]
nadeef_time = [0.0, 4.88841700553894, 9.250931978225708, 13.59175992012024]




openrefine_fscore = 0.415
openrefine_time = 1.5

Plotter(data, real_time, fscore_0,
        runtime_dboost_sec, dboost_fscore,
        nadeef_time, nadeef_fscore,
        openrefine_time, openrefine_fscore, 8)