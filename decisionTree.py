import pandas as pd 
import numpy as np 
import math

# Handle training data so it can be loaded once, then referenced from there
# Make any alteracations to number of features used in training data
# Read in original data and get subset table with columns:
## Is_Home_or_Away
## Is_Opponent_in_AP25_Preseason
## Label
DF_TRAIN = pd.read_csv('Dataset-football-train.txt', sep='\t')
DF_TRAIN = DF_TRAIN[['Is_Home_or_Away', 'Is_Opponent_in_AP25_Preseason', 'Media', 'Label']]



class Tree:
	def __init__(self, observationIDs, features, currLvl=0,subTree={}, bestFeature=None, majorityLabel=None, parentMajorityLabel=None):
		self.observationIDs = observationIDs  # ID
		self.features = features  # Danh sach thuoc tinh
		self.currLvl = currLvl  #cap do hien tai trong cay
		self.subTree = subTree  # cay con
		self.bestFeature = bestFeature  #thuoc tinh tot de chia
		self.majorityLabel = majorityLabel  #nhan pho bien
		self.parentMajorityLabel = parentMajorityLabel # nhan pho bien nhat cua nut cha
		self.setBestFeatureID(bestFeature)

	# predicts using a tree and 
	# observation: [Is_Home_or_Away, Is_Opponent_in_AP25_Preseason, Media]

	def setBestFeatureID(self, feature):
		idx = None
		if feature == 'Is_Home_or_Away':
			idx = 0
		elif feature == 'Is_Opponent_in_AP25_Preseason':
			idx = 1
		else:
			idx = 2
		self.bestFeatureID = int(idx)
# Hàm này được sử dụng để dự đoán kết quả dựa trên cây quyết định.
# Nó sẽ đi xuống cây và dự đoán kết quả cho một quan sát cụ thể.
def predict(tree, obs):
	if tree.bestFeature == None:
		return tree.majorityLabel
	featVal = obs[tree.bestFeatureID]
	if not featVal in tree.subTree:  # val with no subtree
		return tree.majorityLabel
	else:  # recurse on subtree
		return predict(tree.subTree[featVal], obs)
#Hàm này dùng để hiển thị cây quyết định. Nó sẽ in ra cây theo cấp độ và
# hiển thị thuộc tính được chọn tại mỗi nút.
def displayDecisionTree(tree):
	print('\t'*tree.currLvl + '(lvl {}) {}'.format(tree.currLvl,tree.majorityLabel))
	if tree.bestFeature == None:  #nut la
		return

	print('\t'*tree.currLvl + '{}'.format(tree.bestFeature) + ': ')
	for [val, subTree] in sorted(tree.subTree.items()):
		print('\t'*(tree.currLvl+1) + 'choice: {}'.format(val))
		displayDecisionTree(subTree)
#Hàm Entropy tính entropy của tập dữ liệu dựa trên một danh sách các nhãn.
def Entropy(ns): #ns: danh sach chua so lan xuat hien cua tung nhan
	entropy = 0.0
	total = sum(ns)
	for x in ns:
		entropy += -1.0*x/total*math.log(1.0*x/total, 2)
	return entropy

# Hàm IG tính Information Gain (IG) dựa trên các quan sát và một thuộc tính cụ thể.
def IG(observationIDs, feature):
	# get smaller dataframe
	df = DF_TRAIN.loc[list(observationIDs)]

	# điền số lượng Thắng/Thua cho từng danh mục
	labelCountDict = {}
	valueLabelCountDict = {}
	#xay dung bang tan suat cho so lan xuat hien cua nhan trong tung gia tri feature
	for index, row in df.iterrows():
		label = row['Label']
		if not label in labelCountDict:
			labelCountDict[label] = 0 # không tìm thấy nhãn cụ thể này nên hãy chèn số 0
		labelCountDict[label] += 1
		featureValue = row[feature]
		if not featureValue in valueLabelCountDict:
			valueLabelCountDict[featureValue] = {} # không tìm thấy giá trị tính năng cụ thể này nên hãy chèn lệnh trống
		if not label in valueLabelCountDict[featureValue]:
			valueLabelCountDict[featureValue][label] = 0 # không tìm thấy nhãn cụ thể này cho giá trị tính năng này nên hãy chèn 0
		valueLabelCountDict[featureValue][label] += 1

	ns = []
	for [label,count] in labelCountDict.items():
		ns.append(count)

	H_Y = Entropy(ns)

	H_Y_X = 0.0
	for [featureValue, labelCountDict] in valueLabelCountDict.items():
		nsHYX = []
		for [label,count] in labelCountDict.items():
			nsHYX.append(count)
		H_Y_X += 1.0*sum(nsHYX)/len(df)*Entropy(nsHYX)
	return H_Y - H_Y_X
#Hàm GR tính Gain Ratio (GR) dựa trên các quan sát và một thuộc tính cụ thể.
def GR(observationIDs, feature):
	ig = IG(observationIDs, feature)
	if ig == 0:
		return 0
	df = DF_TRAIN.loc[list(observationIDs)]
	valueLabelDict = {}
	for index, row in df.iterrows():
		label = row['Label']
		featureValue = row[feature]
		if featureValue not in valueLabelDict:
			valueLabelDict[featureValue] = 0
		valueLabelDict[featureValue] += 1
	ns = []
	for [val,count] in valueLabelDict.items():
		ns.append(count)
	ent = Entropy(ns)
	return float(ig)/ent
#Hàm này dùng để xây dựng cây quyết định. Nó sẽ đệ quy chia dữ liệu thành các cây con dựa
# trên thuộc tính tốt nhất và điều kiện chia.
def fillDecisionTree(tree, decisionTreeAlgo, observationIDs):
	# find the majorityLabel
	df = DF_TRAIN.loc[list(observationIDs)]
	counts = df['Label'].value_counts()
	majorityLabel = df['Label'].value_counts().idxmax()
	if len(counts) > 1:
		if counts['Win'] == counts['Lose']:
			majorityLabel = tree.parentMajorityLabel
	tree.majorityLabel = majorityLabel

	# exit if only one label
	if len(counts) == 1:
		return
	# exit if no features left
	if len(tree.features) == 0: 
		return

	# find best feature
	featureValueDict = {}
	for feature in tree.features: 
		if decisionTreeAlgo == 'ID3':
			metricScore = IG(tree.observationIDs, feature)
		if decisionTreeAlgo == 'C45':
			metricScore = GR(tree.observationIDs, feature)
		featureValueDict[feature] = metricScore
	bestFeature, bestFeatureValue = sorted(featureValueDict.items(), reverse=True)[0]
	# exit if IG or GR is 0
	if bestFeatureValue == 0.0:
		return
	tree.bestFeature = bestFeature

	# find subset of features
	subFeatures = set()
	for feature in tree.features:
		if feature == bestFeature:  #skip the current best feature
			continue
		subFeatures.add(feature)
	
	# find best feature id
	bestFeatureIdx = 0
	if bestFeature == 'Is_Home_or_Away':
		bestFeatureIdx = 0
	elif bestFeature == 'Is_Opponent_in_AP25_Preseason':
		bestFeatureIdx = 1
	else:
		bestFeatureIdx = 2
	
	# find subset of observations
	subObservationsDict = {}
	for obs in tree.observationIDs:
		val = DF_TRAIN.values[obs][bestFeatureIdx]
		if not val in subObservationsDict:
			subObservationsDict[val] = set()
		subObservationsDict[val].add(obs)

	for [val,obs] in subObservationsDict.items():

		tree.subTree[val] = Tree(obs, subFeatures, tree.currLvl + 1, {}, None, None, majorityLabel)
		
		fillDecisionTree(tree.subTree[val], decisionTreeAlgo, observationIDs)

#Hàm này được sử dụng để dự đoán kết quả và thực hiện phân tích của dự đoán. Nó tính toán các số liệu đánh giá như
# accuracy, precision, recall và F1 score
# dựa trên dự đoán và kết quả thực tế.
def predictAndAnalyze(tree, data):
	TP = 0
	FN = 0
	FP = 0
	TN = 0
	for obs in data:
		prediction = predict(tree, obs)
		ground = obs[3]
		if prediction == 'Win' and ground == 'Win':
			TP += 1
		if prediction == 'Win' and ground == 'Lose':
			FP += 1
		if prediction == 'Lose' and ground == 'Win':
			FN += 1
		if prediction == 'Lose' and ground == 'Lose':
			TN += 1
#Accuracy: Là tỷ lệ giữa số lượng dự đoán đúng (TP và TN) và tổng số quan sát.
#Precision: Là tỷ lệ giữa số lượng dự đoán đúng các trận đấu thắng (TP) và tổng số trận đấu được dự đoán là thắng (TP + FP).
#Recall: Là tỷ lệ giữa số lượng dự đoán đúng các trận đấu thắng (TP) và tổng số trận đấu thắng thực tế (TP + FN).
#F1 Score: Là một số đo kết hợp giữa precision và recall, được tính theo công thức F1 = 2 * (precision * recall) / (precision + recall).
	accuracy = float(TP+TN)/len(data)
	precision = float(TP)/(TP + FP)
	recall = float(TP)/(TP + FN)
	F1 = 2*(recall*precision)/(recall+precision)
	print('\nAnalysis:')
	print('accuracy = {}'.format(accuracy))
	print('precision = {}'.format(precision))
	print('recall = {}'.format(recall))
	print('F1 score = {}'.format(F1))


# read in original data and get subset table with columns:
## Is_Home_or_Away
## Is_Opponent_in_AP25_Preseason
## Label
dfTest = pd.read_csv('Dataset-football-test.txt',sep='\t')
dfTest = dfTest[['Is_Home_or_Away','Is_Opponent_in_AP25_Preseason','Media','Label']]

# obsIDs, features, lvl subTree, bestFeature, majority label, parent majority label
initialObservationIDs = set(range(len(DF_TRAIN)))
initialFeatures = set(dfTest.columns.values[:-1])

# prompt user
print("Which decision tree algorithm would you like to use ('ID3' or 'C45)?")
algoChoice = input()
if algoChoice not in {'ID3', 'C45'}:
	print("Invalid algorithm choice. You must choose 'ID3' or 'C45'")
	exit()

print("choice: {}".format(algoChoice))
print("Dữ liệu train:")
print(DF_TRAIN)
print("Dữ liệu test:")
print(dfTest)
MyTree = Tree(initialObservationIDs,initialFeatures)
fillDecisionTree(MyTree, algoChoice, initialObservationIDs)
print('My Decision Tree:')
displayDecisionTree(MyTree)


print('Predicted Labels of Test Data:')
predictAndAnalyze(MyTree, dfTest.values)

