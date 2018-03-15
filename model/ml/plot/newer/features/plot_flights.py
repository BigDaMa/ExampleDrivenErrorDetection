import matplotlib.pyplot as plt
import numpy as np

from ml.plot.old.potential_latex.plotlatex_lib import plot_list_latex


def plot_list(ranges, list_series, list_names):
    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(len(list_series)):
        ax.plot(ranges[i], list_series[i], label=list_names[i])

    ax.set_ylabel('fscore')
    ax.set_xlabel('labels')

    ax.legend(loc=4)

    plt.show()



label_potential = [4, 8, 12, 16, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196, 206, 216, 226, 236, 246, 256, 266, 276, 286]

fscore_metadata_no_svd_absolute_potential = []
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30257029498016214, 0.51009957325746791, 0.65832531280077, 0.78654824867777673, 0.81651549508692367, 0.8303065807459481, 0.84278801606094922, 0.84283069327298898, 0.85875012785107918, 0.84568221968201185, 0.86688511161171411, 0.87603139464681012, 0.8748236242693006, 0.88662041625371657, 0.8912142152023691, 0.88906311250490011, 0.89362545851095476, 0.896097513024673, 0.90286953949314652, 0.91048186785891705, 0.91118056255670932, 0.91922455573505646, 0.92562986947283221, 0.93261945651075862, 0.93468853778806471, 0.94117647058823528, 0.94639999999999991])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.29274084124830391, 0.54129571291603895, 0.68159203980099503, 0.7947661240469811, 0.80162958320275779, 0.82753500212134079, 0.83166291675189208, 0.83218534752661244, 0.86115851822106204, 0.8646123260437375, 0.86871673565937313, 0.87662910338069577, 0.8809594023395263, 0.88582330496037565, 0.89249096767893754, 0.89981282632252979, 0.90447234209493921, 0.90822878593302381, 0.91340754605457597, 0.91459074733096091, 0.91646118946641686, 0.90965669544268213, 0.91497487437185931, 0.92323273242860027, 0.9250075903248659, 0.93063757381643486, 0.93175431553592936])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.5021373610715304, 0.65031878371750851, 0.78150021070375064, 0.80094884488448848, 0.83580922595777951, 0.83631039531478768, 0.84550616800469947, 0.8550278792539896, 0.85014634146341472, 0.86112995445295082, 0.85755926933540616, 0.87508453289537236, 0.87468428210608118, 0.87586753916319648, 0.88460015835312755, 0.88510553564317018, 0.89041231992051673, 0.89673643487749233, 0.89794703957155608, 0.90212088021507519, 0.90809519068007571, 0.91132414619532665, 0.91359504545000503, 0.91750824917508245, 0.91995197118270977, 0.92340042054671068])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.56022635408245758, 0.68972845336481703, 0.81684044855400362, 0.82383048418466576, 0.82875737238153346, 0.84386884571898446, 0.85468157112091059, 0.8637721569408483, 0.86784228254408569, 0.86830909270802414, 0.87416023262809572, 0.88034101288947531, 0.89219929542023158, 0.90444578555622612, 0.90211826121875305, 0.90808416389811741, 0.91134185303514381, 0.91545653471255228, 0.92048960095531895, 0.92490276254113901, 0.9292044209897441, 0.93399701343952213, 0.93365259902224873, 0.93430365865815967, 0.93880389429763567, 0.9436591678279912])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30257029498016214, 0.49273694807996543, 0.63855421686746994, 0.79033257591406714, 0.77163030877858585, 0.77019315188762061, 0.770310272996382, 0.78310855626796383, 0.78870030755211296, 0.81864235055724421, 0.8376125375125042, 0.85922224636106881, 0.88159560996200925, 0.89028607356177303, 0.90607962143812359, 0.90493282740231773, 0.90484482228823104, 0.90696721311475403, 0.90840703466504014, 0.91331582137805623, 0.91403544510083523, 0.92134603689438477, 0.92967409948542024, 0.93084944821302018, 0.93281680089713515, 0.93764669864271866, 0.94009779951100236])

average_metadata_no_svd_absolute_potential = list(np.mean(np.matrix(fscore_metadata_no_svd_absolute_potential), axis=0).A1)


fscore_metadata_no_svd_unigram = []
fscore_metadata_no_svd_unigram.append([0.0, 0.0, 0.0, 0.0, 0.30169140490162233, 0.5203569436698271, 0.66415186176685326, 0.7785599356395817, 0.77560462670872776, 0.77609745851711831, 0.75118279569892465, 0.78299492385786806, 0.80450041288191565, 0.8085106382978724, 0.82605210420841668, 0.84882909815645236, 0.85229639916674926, 0.85742302412237703, 0.85488706775934431, 0.85903614457831323, 0.86421485118749375, 0.8657852564102565, 0.86717191862784204, 0.8719438877755511, 0.87657531506301256, 0.88747134456294241, 0.88802550308826456, 0.89144769215628383, 0.89449901768172879, 0.89851655368896743, 0.90923488072849656])
fscore_metadata_no_svd_unigram.append([0.0, 0.0, 0.0, 0.0, 0.30257029498016214, 0.5503373819163293, 0.69153648869200279, 0.80172837081410198, 0.80184603299293, 0.81147540983606548, 0.81666666666666665, 0.80127884903586766, 0.81568864690461784, 0.82355285443857729, 0.83086995337764114, 0.83712196086183033, 0.84641101278269415, 0.85164835164835151, 0.86108412667573342, 0.85952031326480671, 0.86062140839583134, 0.86350701599450497, 0.86780054116737537, 0.87690504103165301, 0.88251870566514434, 0.88601439408675364, 0.89464390818128314, 0.8991771587191435, 0.90556274256144897, 0.90855991943605241, 0.91008174386920981])
fscore_metadata_no_svd_unigram.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.55476059438635106, 0.68866674695893049, 0.80287032291132754, 0.80341353074234001, 0.81223755248950213, 0.82210673595842498, 0.82269503546099298, 0.82861285645239247, 0.82994417862838921, 0.83664772727272729, 0.84530052440500214, 0.83849648905410978, 0.85572441742654504, 0.85799302277857581, 0.86200224055402797, 0.86622897004513755, 0.86851701782820101, 0.87283533148888204, 0.87631416202844781, 0.87834072851098954, 0.88316750435942137, 0.88973619126133552, 0.89459041731066458, 0.89401394013940139, 0.89692734559654719, 0.89882304356803633])
fscore_metadata_no_svd_unigram.append([0.0, 0.0, 0.0, 0.0, 0.30198446937014672, 0.57471264367816088, 0.70600000000000007, 0.80027055754179144, 0.80524603096341574, 0.81851424172440335, 0.81885521885521884, 0.82060843964671248, 0.81463270724751702, 0.83009565131643814, 0.83360886900709918, 0.84258344854829159, 0.85406367411508799, 0.8622160499950402, 0.86874127269100332, 0.87265917602996246, 0.87615128355869099, 0.88579823702252691, 0.89268867924528306, 0.89863122396349926, 0.89880067400138763, 0.90752083542524353, 0.91137965760322259, 0.91367781155015193, 0.91060666934511847, 0.91517587939698497, 0.92388029724844345])
fscore_metadata_no_svd_unigram.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.54928601136836264, 0.68473847623596851, 0.791974217694147, 0.8073794881967864, 0.81184491706522854, 0.81558858979509852, 0.81577363034316674, 0.82835974480345753, 0.83158223239363349, 0.83776623927095462, 0.84439237405474488, 0.85147043210531903, 0.85252868678283045, 0.86073253833049401, 0.86652587117212243, 0.87209668943773, 0.88002512562814073, 0.88350224636924046, 0.88570831592143751, 0.89243362367509715, 0.8929280267418781, 0.89710644521048788, 0.90281543274243992, 0.90678676162547123, 0.90851933354291092, 0.91025372195428811])

average_metadata_no_svd_unigram = list(np.mean(np.matrix(fscore_metadata_no_svd_unigram), axis=0).A1)


fscore_metadata_no_svd_bigram = []
fscore_metadata_no_svd_bigram.append([0.0, 0.0, 0.0, 0.0, 0.3020815413727852, 0.57131752817722503, 0.70559556786703603, 0.79228063504135371, 0.81416610528341837, 0.8069282708679909, 0.81381322957198443, 0.79365737382112145, 0.83764469048817314, 0.84019736179639515, 0.84624697336561749, 0.84806986206212165, 0.85207100591715967, 0.86401112545942182, 0.85952070590594598, 0.87296094908551658, 0.87629876494804948, 0.88239952484656514, 0.89047427652733124, 0.89893723681572091, 0.91137174910759822, 0.91248851220259364, 0.9202441505595117, 0.92589576547231267, 0.92902176123652636, 0.92902176123652636, 0.93345562009579131])
fscore_metadata_no_svd_bigram.append([0.0, 0.0, 0.0, 0.0, 0.30156815440289508, 0.52189578713968954, 0.64687168610816537, 0.75753968253968251, 0.76068289726234961, 0.79550102249488752, 0.81948131509087196, 0.82926339061400867, 0.8315757758360951, 0.83652605459057072, 0.83159282165669679, 0.83096175194267841, 0.82710038330052837, 0.83377201862725248, 0.83690813435855937, 0.84591290289986876, 0.85281428714758711, 0.85691526436323817, 0.86247366837195316, 0.86641873555131177, 0.88072490889392296, 0.88633435734813204, 0.89620558120859262, 0.89986996098829641, 0.91146881287726367, 0.91546889134586396, 0.91848866498740556])
fscore_metadata_no_svd_bigram.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.57605351170568564, 0.69703243616287103, 0.78897095027080255, 0.79389158576051777, 0.79918864097363074, 0.79380703373321015, 0.818328950030346, 0.81745791754863417, 0.82729691016983842, 0.82375871973738191, 0.83547955083960035, 0.84386391251518833, 0.8568550418051778, 0.86510322709961907, 0.87699696167793784, 0.88001569704699301, 0.87861959539865131, 0.88561250742133379, 0.89719626168224309, 0.90445352196871565, 0.90932772267307882, 0.91048078857372772, 0.91447500501907242, 0.91371170266197888, 0.91617973000201491, 0.92206477732793524])
fscore_metadata_no_svd_bigram.append([0.0, 0.0, 0.0, 0.0, 0.30103448275862071, 0.55153054404491375, 0.68905863850820037, 0.78146351123873647, 0.77785439858142058, 0.77081798084008835, 0.77941329992692343, 0.791326209930861, 0.79380364245342272, 0.78282446865897071, 0.81931984143542669, 0.82341851696690405, 0.83049766056997021, 0.84001697432633149, 0.85212082938638034, 0.87434285125244815, 0.87989259527006092, 0.89213250517598353, 0.89555006180469721, 0.9047814487995075, 0.90825312308007355, 0.91041859508382184, 0.91829430412107571, 0.91531679311051872, 0.91835897435897429, 0.91835897435897429, 0.91730411412032253])
fscore_metadata_no_svd_bigram.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.53175252112292182, 0.62208359326263252, 0.7260447446179823, 0.73942759504485256, 0.74923320994182974, 0.75108247967050368, 0.75020746887966816, 0.78189383943976454, 0.82762106965928617, 0.8401486988847584, 0.86078450745973734, 0.86307874950534236, 0.86784228254408569, 0.8676338169084542, 0.86983030179859777, 0.87448896156991007, 0.88126216576170469, 0.88051867860450761, 0.8893430060284051, 0.89427402862985683, 0.89765238879736409, 0.89766718506998455, 0.9005289907685925, 0.90170895908855531, 0.90813377374071025, 0.91438037063878241])

average_metadata_no_svd_bigram = list(np.mean(np.matrix(fscore_metadata_no_svd_bigram), axis=0).A1)


#####
#experiments lstm

fscore_metadata_deep = []
fscore_metadata_deep.append([0.0, 0.0, 0.0, 0.0, 0.30286305622628495, 0.56404072883172562, 0.70039886039886035, 0.79317150187487673, 0.8126303811871799, 0.81840375586854464, 0.81840375586854464, 0.81445120753976041, 0.83351193375547983, 0.8495184590690209, 0.85279187817258884, 0.86123725905742254, 0.8771788413098236, 0.88618133820692468, 0.89577635976906711, 0.90080910997902308, 0.90178482401036997, 0.90511095491515203, 0.91088510723995575, 0.91217727180321206, 0.91310541310541304, 0.92026780279975662, 0.92050466714534807, 0.92616899097620997, 0.9367062887331099, 0.94470934361367553, 0.9459016393442623])
fscore_metadata_deep.append([0.0, 0.0, 0.0, 0.0, 0.30286305622628495, 0.54376473852129292, 0.60384331116038437, 0.70368358144083454, 0.72283766510438852, 0.72283766510438852, 0.79597821533305402, 0.83967499009116142, 0.85494124446156816, 0.85376260667183868, 0.86787641000490434, 0.87216271985850435, 0.86785886901884968, 0.86876408899343327, 0.87816455696202533, 0.88851216593538107, 0.89502762430939231, 0.90225106469276017, 0.90732604475324419, 0.91012279434526888, 0.90984113884877249, 0.91634121274409031, 0.92770962181891925, 0.92893298238509814, 0.93395937531897533, 0.93586454592098511, 0.93913934426229495])
fscore_metadata_deep.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.57386666666666664, 0.7108698163336028, 0.80556636823666805, 0.80614726193950004, 0.81421036919096423, 0.80708661417322836, 0.83131907860758192, 0.82779085234830652, 0.83538182550876483, 0.83900364520048598, 0.85403225806451621, 0.85746034953025563, 0.80909948631932072, 0.81760741364785172, 0.83780654667929699, 0.89707801988151425, 0.90430719219829325, 0.91292929292929292, 0.91874366767983784, 0.91815161582603444, 0.92327569071755122, 0.92929091453076695, 0.93263623567263221, 0.93745607870695713, 0.94377712728373875, 0.94725118963247945])
fscore_metadata_deep.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.53295853829936757, 0.67496074405121387, 0.77265170407315042, 0.77288768943325725, 0.79156664231291096, 0.81710213776722085, 0.82139917695473252, 0.82466086252277782, 0.84095771777890982, 0.86947930046438082, 0.88073394495412838, 0.88545398710205747, 0.89066888588369919, 0.89403838301347494, 0.8964656964656964, 0.90251506963209316, 0.91848825331971407, 0.92430116302795351, 0.93499082942734868, 0.93929517213281732, 0.93719110378912684, 0.93876710919007933, 0.94211452832142495, 0.94210090984284534, 0.94591531884207258, 0.95504699632202705])
fscore_metadata_deep.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.55440696409140378, 0.61758548499651089, 0.72946954813359521, 0.73010889826351422, 0.78305785123966942, 0.7919183503436783, 0.78142318776324526, 0.81455578625286862, 0.82020968101717606, 0.82426916712631004, 0.86193464618048032, 0.86548058147103502, 0.87454969273151106, 0.87771570453134695, 0.88695471971816386, 0.89567966280295042, 0.90444793301936155, 0.89703867636210344, 0.90634820867379018, 0.91750208855472015, 0.92112646783747265, 0.92851222544550349, 0.93580068231158897, 0.93733250876108021, 0.9412974195538194, 0.94702326057997754])

average_metadata_deep = list(np.mean(np.matrix(fscore_metadata_deep), axis=0).A1)



fscore_metadata_only = []
fscore_metadata_only.append([0.0, 0.0, 0.0, 0.0, 0.24586704185719307, 0.46066927422859627, 0.54864555513162916, 0.70195720515160731, 0.72108978820721881, 0.72725490196078424, 0.71999214839532821, 0.71733771569433036, 0.72255115820089344, 0.74683418087546116, 0.78794992175273859, 0.80997263605959258, 0.85083415112855754, 0.86156612065521365, 0.88166034723608067, 0.88904536766924858, 0.88806120394604393, 0.88289380620390023, 0.885578189926016, 0.88902340597255847, 0.89393330654881475, 0.89394702386947322, 0.891063404892661, 0.89647554975399146, 0.9027291812456264, 0.90502793296089379, 0.90976837865055393])
fscore_metadata_only.append([0.0, 0.0, 0.0, 0.0, 0.24921246062303115, 0.5249143158449775, 0.60202302057900248, 0.70280296784830998, 0.75309283709325292, 0.77798353909465034, 0.78953698135898986, 0.80752475247524758, 0.8369905956112853, 0.83547557840616959, 0.84732214228617098, 0.84543537210003017, 0.85796583234008728, 0.86864069952305245, 0.87548367893640233, 0.88, 0.8896531394320526, 0.89142183354623161, 0.88777318369757841, 0.89183093066588215, 0.89622733953944145, 0.89825524406979007, 0.90167772142021063, 0.90217604391295825, 0.9028459273797842, 0.90654662882865245, 0.91008386970938182])
fscore_metadata_only.append([0.0, 0.0, 0.0, 0.0, 0.12004573170731707, 0.4023408924652524, 0.50424510889627172, 0.63525674960296452, 0.70984507746126946, 0.73521011831905358, 0.80134751026423834, 0.81059273818144206, 0.80884573894282641, 0.82992042440318303, 0.85521031494398647, 0.87389672314091504, 0.88496294040004064, 0.88572294634737792, 0.890027116601386, 0.89591040895910412, 0.90167575206945283, 0.90151975683890584, 0.9022873194221509, 0.90544354838709684, 0.90844999496424617, 0.91146413032480011, 0.91560670300827773, 0.9174644764688098, 0.91126314714592038, 0.90844135959987748, 0.91781376518218627])
fscore_metadata_only.append([0.0, 0.0, 0.0, 0.0, 0.28853016142735766, 0.46305841924398633, 0.53835565010636965, 0.66988199076449451, 0.7025210084033614, 0.70643827525103364, 0.73637012126770607, 0.78240312467879525, 0.81643105920917847, 0.81767051845792016, 0.83581786837034777, 0.85671402841705036, 0.86661976077997793, 0.86941718405768076, 0.87983049137322178, 0.88993679141165849, 0.89288893284541593, 0.89450310865488991, 0.89830841856805665, 0.90153301886792447, 0.89929879236462795, 0.90306867998051632, 0.90146872872288686, 0.90672408730547116, 0.90939399882329874, 0.9075053889868705, 0.91019488786602687])
fscore_metadata_only.append([0.0, 0.0, 0.0, 0.0, 0.25838984524430531, 0.53364269141531329, 0.58284667353715791, 0.7038267970753016, 0.69524191929331447, 0.75601103665746949, 0.77553024351924582, 0.80110602593440117, 0.82958391399562026, 0.87389380530973448, 0.8790118457021362, 0.86826162850395328, 0.87313202102442544, 0.87450899317758946, 0.87333196468897556, 0.87634024303073621, 0.88084495488105008, 0.88491048593350374, 0.89032655576093656, 0.90604026845637575, 0.91054280804609322, 0.91241830065359475, 0.91544600464693393, 0.91894619965680835, 0.92309249139850236, 0.92416725726435156, 0.92952342487883688])

average_metadata_only = list(np.mean(np.matrix(fscore_metadata_only), axis=0).A1)



######




ranges = [label_potential, label_potential, label_potential, label_potential, label_potential]
list = [average_metadata_no_svd_unigram, average_metadata_no_svd_bigram, average_metadata_only, average_metadata_no_svd_absolute_potential, average_metadata_deep]
names = ["Unigrams", "Unigrams + Bigrams", "Metadata", "Unigrams + Metadata", "LSTM + Metadata"]


plot_list_latex(ranges, list, names, "Flights", x_max=200)
plot_list(ranges, list, names)