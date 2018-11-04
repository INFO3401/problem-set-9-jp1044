#worked with Marissa Kelley, Taylor Lawrence, and Hannah Weber
#Problem set 9 - Jacob Paul

import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd


class AnalysisData:
    def __init__ (self):
        self.dataset = pd.DataFrame()
        self.xs = []


    def openCSV(self, filename):
        self.dataset = pd.read_csv(filename)
        self.xs = [val for val in self.dataset.columns.values if val != "competitorname"]

#Problem 2
class LinearAnalysis:
    def __init__ (self, _targetY):
        self.targetY = _targetY
        self.bestX = ""
    #Problem 3
    def runSimpleAnalysis(self, data):
        best_r2 = -1
        best_var = ""
        for column in data.xs:
            if column != self.targetY:
                #Set up indenpendent variable
                independent_var = data.dataset[column].values
                independent_var = independent_var.reshape(len(independent_var),1)
                # Do regression
                regr = LinearRegression()
                regr.fit(independent_var, data.dataset[self.targetY])
                pred = regr.predict(independent_var)
                r_score = r2_score(data.dataset[self.targetY],pred)
                #If current r_score is better than our previous best, then the current r_score is new best
                if r_score > best_r2:
                    best_r2 = r_score
                    best_var = column
        self.bestX = best_var
        print(best_var, best_r2)


#Problem 2
class LogisticAnalysis:
    def __init__ (self, targetY):
        self.targetY = _targetY
        self.bestX = ""


#Problem 1
analysis_data = AnalysisData()
analysis_data.openCSV('candy-data.csv')

#Problem3
line_analysis = LinearAnalysis('sugarpercent')
line_analysis.runSimpleAnalysis(analysis_data)
