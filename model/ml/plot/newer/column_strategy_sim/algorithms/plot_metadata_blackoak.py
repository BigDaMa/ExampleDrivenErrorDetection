import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform_latex


labels_all = [4, 8, 12, 16, 20, 24, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498]

svm_sim = []
svm_sim.append([4.7921632357932555e-05, 0.0003837740638193715, 0.002143898772765056, 0.09260833577516012, 0.17616144758308558, 0.24522329227966672, 0.25940034986163746, 0.2799663974232395, 0.3048241385183045, 0.31232592916048446, 0.3894944913829132, 0.5062615573580581, 0.5928294095336237, 0.6101403950342345, 0.6372350359754047, 0.6392010240935433, 0.6914915578028179, 0.6991454625331245, 0.7095138799766672, 0.7162890905950815, 0.7188112390943383, 0.725084257890581, 0.7304833004104865, 0.7431472124211329, 0.7483150187010388, 0.7599248260103307, 0.773522746982308, 0.7792647484571472, 0.7809572406365687, 0.7842781334926323, 0.7898756419066761, 0.7927598232217469, 0.7994128057158596, 0.7994773302225656, 0.8098414441136939, 0.8164414019585122, 0.821101950664846, 0.8297043848913235, 0.8304133249891039, 0.8336137707818695, 0.8361702871018741, 0.8421006353166207, 0.8455294095095679, 0.8465371844434408, 0.8445581459634054, 0.8485185803289965, 0.8528459109904579, 0.8571045402610153, 0.8620312259451242, 0.8742565361819474, 0.8763125327363588, 0.8787429941621543, 0.8848172754998872, 0.8868491630992585])
average_svm_sim = list(np.mean(np.matrix(svm_sim), axis=0).A1)

bayes_sim = []
bayes_sim.append([6.552102581858636e-05, 0.0008619469594944231, 0.003215448338052865, 0.17024961159526636, 0.25746959386302637, 0.4132951165400498, 0.43926426357590653, 0.4403544826312489, 0.47368806787584716, 0.4997510464482563, 0.512814448131494, 0.5331969928780339, 0.5301992029681861, 0.5263856038718321, 0.530631555060575, 0.5411347730247776, 0.5519908659617453, 0.5573867761187324, 0.5652939205528248, 0.5720873612121415, 0.5736341067568738, 0.5825047578593658, 0.5871304185360657, 0.5925859116958451, 0.5996479648743657, 0.6098653643674774, 0.6175019743693947, 0.6193942666561181, 0.6259660231881993, 0.6277694671233298, 0.6337380553493644, 0.6410585973938103, 0.6428749660134233, 0.6472859377422016, 0.663164957910616, 0.6651861550637823, 0.6688289701440795, 0.6723443220166114, 0.6766130640259344, 0.6804076946705961, 0.6821300720419801, 0.6843737748079927, 0.6883080472181118, 0.6911481502048953, 0.6930975935944038, 0.6930537168727847, 0.6934486129827945, 0.6940879828588067, 0.7058552593653227, 0.7085439589081342, 0.7107297661124357, 0.7146101565809159, 0.7177420125497052, 0.7228476634286406])
average_bayes_sim = list(np.mean(np.matrix(bayes_sim), axis=0).A1)

trees_sim = []
trees_sim.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14240343779747952, 0.5741965011024701, 0.6522479213872837, 0.7754172263354274, 0.8348535468119946, 0.9201583523198711, 0.9467868300437523, 0.9557015103847423, 0.9699006951278741, 0.9746786998563955, 0.9762495029966646, 0.976545131810877, 0.9790016833913986, 0.9784384381945717, 0.980207552681498, 0.9776896583207882, 0.9821551159400732, 0.9837952497957545, 0.9829444718035221, 0.9851874777131442, 0.9842950322383803, 0.9858819102302974, 0.986225954973888, 0.986627587462511, 0.9869414419184146, 0.9870350877016698, 0.9873929884459466, 0.9878255751809604, 0.9886482204112863, 0.988889526263099, 0.9890766571575952, 0.9890557075074599, 0.9898389569861441, 0.9898419789080177, 0.9899748656743992, 0.9912639400969289, 0.991396010646308, 0.9915440908783468, 0.9920261300132875, 0.9921330040192424, 0.992160054515851, 0.9921668779730888, 0.992212602756771, 0.9922987806783])
average_trees_sim = list(np.mean(np.matrix(trees_sim), axis=0).A1)



ranges = [labels_all,
		  labels_all,
          labels_all
		  ]
list = [average_trees_sim,
		average_svm_sim,
		average_bayes_sim

		]
names = [
		 "Gradient Tree Boosting",
		 "Linear SVM",
		 "Multinomial Naive Bayes"
		 ]

plot_list_latex(ranges, list, names, "Address", x_max=200)
plot_list(ranges, list, names, "Address", x_max=200, end_of_round=98)
plot_integral(ranges, list, names, "Address", x_max=150, x_min=98)
#plot_end(ranges, list, names, "Address", x_max=150, x_min=98)
#plot_integral_latex(ranges, list, names, "Address", x_max=350)
#plot_outperform_latex(ranges, list, names, "Address",0.904, x_max=350)