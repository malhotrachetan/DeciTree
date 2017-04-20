'''
Author: Chetan Malhotra

Made for 5th semester course: Design and Analysis(CSE-311) project(Project Based Learning)

This app.py file contains both the gui frontend code and the code that generates that Decision tree for given data set
'''

#imports:

from tkinter import *

import sys
import csv
import collections


#Implementation of Decision Tree using ID3 algorithm


class DecisionTree:
    #Binary tree implementation with true and false branch.

    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None, results=None):
        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results  # None for nodes, not None for leaves


def divideSet(rows, column, value):
    splittingFunction = None
    if isinstance(value, int) or isinstance(value, float):  # for int and float values
        splittingFunction = lambda row: row[column] >= value
    else:  # for strings
        splittingFunction = lambda row: row[column] == value
    list1 = [row for row in rows if splittingFunction(row)]
    list2 = [row for row in rows if not splittingFunction(row)]
    return (list1, list2)


def uniqueCounts(rows):
    results = {}
    for row in rows:
        r = row[-1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results


def entropy(rows):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = uniqueCounts(rows)

    entr = 0.0
    for r in results:
        p = float(results[r]) / len(rows)
        entr -= p * log2(p)
    return entr


def gini(rows):
    total = len(rows)
    counts = uniqueCounts(rows)
    imp = 0.0

    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def variance(rows):
    if len(rows) == 0: return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)

    variance = sum([(d - mean) ** 2 for d in data]) / len(data)
    return variance


def growDecisionTreeFrom(rows, evaluationFunction=entropy):
    """Grows and then returns a binary decision tree.
    evaluationFunction: entropy or gini"""

    if len(rows) == 0: return DecisionTree()
    currentScore = evaluationFunction(rows)

    bestGain = 0.0
    bestAttribute = None
    bestSets = None

    columnCount = len(rows[0]) - 1  # last column is the result/target column
    for col in range(0, columnCount):
        columnValues = [row[col] for row in rows]

        for value in columnValues:
            (set1, set2) = divideSet(rows, col, value)

            # Gain -- Entropy or Gini
            p = float(len(set1)) / len(rows)
            gain = currentScore - p * evaluationFunction(set1) - (1 - p) * evaluationFunction(set2)
            if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                bestGain = gain
                bestAttribute = (col, value)
                bestSets = (set1, set2)

    if bestGain > 0:
        trueBranch = growDecisionTreeFrom(bestSets[0])
        falseBranch = growDecisionTreeFrom(bestSets[1])
        return DecisionTree(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch,
                            falseBranch=falseBranch)
    else:
        return DecisionTree(results=uniqueCounts(rows))


def prune(tree, minGain, evaluationFunction=entropy, notify=False):
    """Prunes the obtained tree according to the minimal gain (entropy or Gini). """
    # recursive call for each branch
    if tree.trueBranch.results == None: prune(tree.trueBranch, minGain, evaluationFunction, notify)
    if tree.falseBranch.results == None: prune(tree.falseBranch, minGain, evaluationFunction, notify)

    # merge leaves (potentionally)
    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        tb, fb = [], []

        for v, c in tree.trueBranch.results.items(): tb += [[v]] * c
        for v, c in tree.falseBranch.results.items(): fb += [[v]] * c

        p = float(len(tb)) / len(tb + fb)
        delta = evaluationFunction(tb + fb) - p * evaluationFunction(tb) - (1 - p) * evaluationFunction(fb)
        if delta < minGain:
            if notify: print('A branch was pruned: gain = %f' % delta)
            tree.trueBranch, tree.falseBranch = None, None
            tree.results = uniqueCounts(tb + fb)


def classify(observations, tree, dataMissing=False):
    """Classifies the observationss according to the tree.
    dataMissing: true or false if data are missing or not. """

    def classifyWithoutMissingData(observations, tree):
        if tree.results != None:  # leaf
            return tree.results
        else:
            v = observations[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            else:
                if v == tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
        return classifyWithoutMissingData(observations, branch)

    def classifyWithMissingData(observations, tree):
        if tree.results != None:  # leaf
            return tree.results
        else:
            v = observations[tree.col]
            if v == None:
                tr = classifyWithMissingData(observations, tree.trueBranch)
                fr = classifyWithMissingData(observations, tree.falseBranch)
                tcount = sum(tr.values())
                fcount = sum(fr.values())
                tw = float(tcount) / (tcount + fcount)
                fw = float(fcount) / (tcount + fcount)
                result = collections.defaultdict(
                    int)  
                for k, v in tr.items(): result[k] += v * tw
                for k, v in fr.items(): result[k] += v * fw
                return dict(result)
            else:
                branch = None
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
                else:
                    if v == tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
            return classifyWithMissingData(observations, branch)

    # function body
    if dataMissing:
        return classifyWithMissingData(observations, tree)
    else:
        return classifyWithoutMissingData(observations, tree)


def plot(decisionTree):
    """Plots the obtained decision tree. """

    def toString(decisionTree, indent=''):
        if decisionTree.results != None:  # leaf node
            return str(decisionTree.results)
        else:
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = 'Column %s: x >= %s?' % (decisionTree.col, decisionTree.value)
            else:
                decision = 'Column %s: x == %s?' % (decisionTree.col + 1, decisionTree.value)
            trueBranch = indent + 'yes -> ' + toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no  -> ' + toString(decisionTree.falseBranch, indent + '\t\t')
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))


def loadCSV(file):
    """Loads a CSV file and converts all floats and ints into basic datatypes."""

    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'rt'))
    return [[convertTypes(item) for item in row] for row in reader]

def main_func(getvalue):
    # Select the example you want to classify


    # All examples do the following steps:
    # 	1. Load training data
    # 	2. Let the decision tree grow
    # 	4. Plot the decision tree
    # 	5. classify without missing data
    # 	6. Classifiy with missing data
    # 	7. Prune the decision tree according to a minimal gain level
    # 	8. Plot the pruned tree


        if getvalue == 1:

        # the smaller examples
            trainingData = loadCSV('mobile.csv')


            decisionTree = growDecisionTreeFrom(trainingData)
            decisionTree = growDecisionTreeFrom(trainingData, evaluationFunction=gini)  # with gini
            plot(decisionTree)



            print("\nClassifying without missing data(All the attribute values provided):\n")

            print(classify(['low','young','masters','married','service'], decisionTree, dataMissing=False))

            print("\nClassifying with missing data(Some of the attribute values provides): \n")
            print(classify([None, 'young', None, None, 'business'], decisionTree, dataMissing=True)) # no longer unique



        elif getvalue==2:
            # the bigger example
            trainingData = loadCSV('fishiris.csv')
            decisionTree = growDecisionTreeFrom(trainingData)
            plot(decisionTree)

            prune(decisionTree, 0.5, notify=True)  # notify, when a branch is pruned (one time in this example)
            plot(decisionTree)
            print("\n")
            print("\nClassifying without missing data(All the attribute values provided):\n")
            print(classify([5.4, .92, 1.2, 0.1], decisionTree))  # dataMissing=False is the default setting
            print("\nClassifying with missing data(Some of the attribute values provides): \n")

            print(classify([None, None, None, 1.5], decisionTree, dataMissing=True)) #data missing=true






#GUI app code begins:

root=Tk()

root.geometry("1500x1500")

def frame_dataset1_about():
    windowdataset1about = Toplevel()
    windowdataset1about.geometry('1500x1500')
    labeldataset1about = Label(windowdataset1about,
                               text='Mobile Usage data set is a simple as well as sample data set.\nIt consists of 5 attributes: \nColumn 1: Salary\n2.Age\n3.Higest degree\n4.Marital status\n5.Job profile\nAnd the last is the MOBILE USAGE which is predicted by the decision tree',
                               background='light green', anchor=CENTER)
    labeldataset1about.config(font=('comic sans', 25))
    labeldataset1about.pack()


def frame_dataset1():
    window3 = Toplevel()
    window3.geometry('1500x1500')
    main_func(1)

    label4= Label(window3, text="The Decision tree for MOBILE USAGE data set can be viewed in the console/terminal\nClick on ABOUT to read more about this data-set", anchor=CENTER,
                   height=8, width=74, background='light green')
    label4.config(font=('comic sans', 38, 'bold'))
    label4.pack()
    button_dataset1_about=Button(window3,text='ABOUT',height=20,width=40,command=main_func(1))
    button_dataset1_about.config(font=('times', 30, 'bold'))
    button_dataset1_about.pack()


def frame_dataset2_about():
    windowdataset2about=Toplevel()
    windowdataset2about.geometry('1500x1500')
    labeldataset2about=Label(windowdataset2about,
                        text='The Iris flower data set is a multivariate data set introduced by Ronald Fisher \nin his 1936 paper The use of multiple measurements in taxonomic problems \nas an example of linear discriminant analysis.\nThe data set consists of 50 samples from each of three species of Iris\n (Iris setosa, Iris virginica and Iris versicolor).\n Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres. \nBased on the combination of these four features\nVisit https://en.wikipedia.org/wiki/Iris_flower_data_set for more information',
                        background='light green',anchor=CENTER)
    labeldataset2about.config(font=('comic sans',25))
    labeldataset2about.pack()






def frame_dataset2():
    window4 = Toplevel()
    window4.geometry('1500x1500')
    main_func(2)

    label5 = Label(window4,
                   text="The Decision tree for IRIS data set can be viewed in your console/terminal\nClick on ABOUT to read more about this data-set",
                   anchor=CENTER,
                   height=8, width=74, background='light green')
    label5.config(font=('comic sans', 38, 'bold'))
    label5.pack()
    button_dataset2_about = Button(window4, text='ABOUT', height=20, width=40,command=frame_dataset2_about)
    button_dataset2_about.config(font=('times', 30, 'bold'))
    button_dataset2_about.pack()



#title
root.wm_title("DeciTree")

def frame_id3():
    window2=Toplevel()
    window2.geometry("1500x1500")
    label3 = Label(window2, text="Please choose a data-set.", anchor=CENTER,
                   height=8, width=74, background='light green')
    label3.config(font=('comic sans', 38, 'bold'))
    label3.pack()



    button_dataset1 = Button(window2, text='MOBILE USAGE DATA SET',height=20,width=40,command=frame_dataset1)
    button_dataset1.config(font=('times',30,'bold'))
    button_dataset2 = Button(window2, text='IRIS DATA SET',height=20,width=40,command=frame_dataset2)
    button_dataset2.config(font=('times', 30,'bold' ))



    button_dataset1.pack(side=LEFT)
    button_dataset2.pack(side=RIGHT)

def frame_id3_about():
    windowid3about = Toplevel()
    windowid3about.geometry("1500x1500")
    labelid3about=Label(windowid3about,text='In decision tree learning,\n ID3 (Iterative Dichotomiser 3) is an algorithm invented by Ross Quinlan[1] used to generate a decision tree from a dataset.\n Visit: https://en.wikipedia.org/wiki/ID3_algorithm',
                        background='light green',anchor=CENTER)
    labelid3about.config(font=('comic sans',25))
    labelid3about.pack()



def frame2():
    #windows for algo selection

    window1=Toplevel()
    window1.geometry("1500x1500")
    label2=Label(window1,text="Decision tree will be implemented using ID3 algorithm.\n\n\nClick on the ID3 button to continue.\n\nClick on ABOUT to know about ID3.",anchor=CENTER,
                 height=10,width=74,background='light green')
    label2.config(font=('comic sans',38,'bold'))
    label2.pack()
    #buttons for algorithm selection
    #button for id3
    button_id3=Button(window1,text="ID3",height=30,width='30',command=frame_id3)
    button_id3.config(font=('times',30,'bold'))
    button_id3.pack(side=LEFT)
    button_id3_about=Button(window1,text="ABOUT",height="30",width=30,border='20px',command=frame_id3_about)
    button_id3_about.config(font=('times',30,'bold'))
    button_id3_about.pack(side=RIGHT)



def frame1():
    # label(text displayed on the frame1)

    label1 = Label(root,text="Welcome to DeciTree!\n\nThis is a desktop GUI app written in Python \nthat builds a predictive model \nusing a Decision tree algorithm called ID3\n\nClick NEXT to continue ",
                   anchor=CENTER, height=14, width=70, background='light green')
    label1.config(font=('comic sans', 38, 'bold'))
    label1.pack()

    # button(next button on frame1)
    NEXTbutton1 = Button(root, text="NEXT", height=30,width=30, command=frame2)
    NEXTbutton1.config(font=('times', 30,'bold'))
    NEXTbutton1.pack()

frame1()



root.mainloop()
