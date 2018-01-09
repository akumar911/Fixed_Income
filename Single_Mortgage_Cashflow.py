__author__ = 'Aviral Kumar'
__date__ = 12 / 15 / 2017
__version__ = '1.0'
__copyright__ = 'The Oakleaf Group, LLC.'


import pandas as pd
from datetime import datetime
from datetime import timedelta
from collections import  OrderedDict
import warnings
import numpy as np
import decimal
import sys
import os

######################### ALL THE COMMON FUNCTIONS GO HERE ################################

def prep_cashflow_months():
        """
        The idea behind this function is to start the Amortization process from the beginning of Next Month.

        :return: A list of all the amortization months
        """

        today = datetime.today().date()
        if today.month == 12:
            today = today.replace(month=1)
            today = today.replace(year=today.year + 1)
        else:
            today = today.replace(month=today.month + 1)
        today = today.replace(day=1)
        rge = pd.date_range(start=today, periods=360, freq='M')
        return rge

######################### END ################################


warnings.filterwarnings("ignore")



class PassThroughMBS:
    """
    This class will create a simple Pass Through MBS Security

    """

    def __init__(self, **kwargs):
        """
        Use of kwargs to initialize the class as well as empty dataframes and lists
        :param kwargs:
        :type kwargs  :Dict
        Term : What is the WAM of the MBS
        Original_Mortgage _Balance : The underlying mortgage balance of the entire pool
        PSA (prepayment multiplier)
        Mortgage Rate : WAC
        Pass-Through Rate : This will be strictly lower than the WAC because it takes the fees associated with servicing the mortgage into account as well
        Seasoning : How Old is the loan
        """

        self.n = kwargs['Term']
        self.mortgage_balance = float(kwargs['Original_mortgage_balance'])
        self.mortgage_rate = float(kwargs['mortgage_rate']/12)
        self.passthrough_rate = float(kwargs['pass_through_rate']/12)
        self.PSA = kwargs['prepayment_multiplier']
        self.seasoning = kwargs['Seasoning']
        self.monthly_payment = round((self.mortgage_balance*self.mortgage_rate)*((1 + self.mortgage_rate)**(self.n - self.seasoning))/(((1 + self.mortgage_rate)**(self.n - self.seasoning)) -1),3)
        self.rge = prep_cashflow_months()

        self.psa_parameters = pd.DataFrame(data = [[1,.002],[30,.06],[360,.06]], columns = ['Time','CPR Rate'])


        self.prep_dataframe()


    def get_cpr(self,x):
        month = x['Months']

        if (month + self.seasoning <= self.psa_parameters.loc[1,'Time']):
            CPR = self.psa_parameters.loc[0,'CPR Rate'] + (self.psa_parameters.loc[1,'CPR Rate'] - self.psa_parameters.loc[0,'CPR Rate'])*(month+ self.seasoning -1)/ (self.psa_parameters.loc[1,'Time']
                                                                                                            - self.psa_parameters.loc[0,'Time'])
        else:
            CPR = self.psa_parameters.loc[1,'CPR Rate'] + (self.psa_parameters.loc[2,'CPR Rate'] - self.psa_parameters.loc[1,'CPR Rate'])*(month + self.seasoning - self.psa_parameters.loc[1,'Time'])  / (self.psa_parameters.loc[2,'Time']
                                                                                                            - self.psa_parameters.loc[1,'Time'])

        return CPR

    def prep_dataframe(self):

        """
        An important thing to note here is that we can;t avoid FOR loop here. Unlike in the case of a Single Mortgage Amortizer where the beginning monthly balance was a function of Initila Mortgage Balance.
        Here, since we have prepayment, we cannot do it the same way as the beginning monthly balance depends upon the principal prepayed in the previous month.
        :return: A clean Data Frame with the Amortization table
        """

        # df = pd.DataFrame(index= pd.Series(self.rge).apply(lambda x : x.replace(day = 1)), columns = ['Months','Conditional Payment Rate (CPR)','Single-Monthly Mortality (SMM)','Beginning Monthly Balance',
        #                                                                                               'Monthly Payment','Monthly Interest Paid in by Mortgage Holders','Monthly Interest Paid out to Investors','Scheduled Principal Payment',
        #                                                                                               'Prepayment','Total Principal Payment','Ending Mortgage Balance'])
        #
        # df['Months'] = range(1, 361)
        # df.index.name = 'Monthly Payments'
        # df['Conditional Payment Rate (CPR)'] = df.apply(lambda x : self.get_cpr(x),axis = 1)
        # df['Single-Monthly Mortality (SMM)'] = 1 - (1 - df['Conditional Payment Rate (CPR)'])**(0.083)
        # df['Single-Monthly Mortality (SMM)']  = df['Single-Monthly Mortality (SMM)'].apply(lambda x : round(x,4))
        # df['Monthly Payment'] = self.monthly_payment
        # df['Monthly Payment'] = df['Monthly Payment']
        # df['Beginning Monthly Balance'] = self.mortgage_balance*((1 + self.mortgage_rate)**(df['Months']-1)) - self.monthly_payment*((((1 + self.mortgage_rate)**(df['Months']-1))-1)/self.mortgage_rate)
        # df['Monthly Interest Paid in by Mortgage Holders'] = df['Beginning Monthly Balance'] * self.mortgage_rate
        # df['Monthly Interest Paid out to Investors'] = df['Beginning Monthly Balance'] * self.passthrough_rate
        # df['Monthly Interest Paid in by Mortgage Holders'] = df['Monthly Interest Paid in by Mortgage Holders'].apply(lambda x: round(x,2))
        # df['Monthly Interest Paid out to Investors'] = df['Monthly Interest Paid out to Investors'].apply(lambda x : round(x,3))
        # df['Scheduled Principal Payment'] = df['Monthly Payment'] - df['Monthly Interest Paid in by Mortgage Holders']
        # df['Scheduled Principal Payment'] = df['Scheduled Principal Payment'].apply(lambda x : round(x,3))
        # df['Prepayment'] = (df['Beginning Monthly Balance'] - df['Scheduled Principal Payment']) * df['Single-Monthly Mortality (SMM)']
        # df['Prepayment'] = df['Prepayment'].apply(lambda x  : round(x,3))
        # df['Total Principal Payment'] = df['Scheduled Principal Payment'] + df['Prepayment']
        # df['Total Principal Payment'] = df['Total Principal Payment'].apply(lambda x : round(x,3))
        # df['Ending Mortgage Balance'] = df['Beginning Monthly Balance'] - df['Total Principal Payment']
        # df['Ending Mortgage Balance'] = df['Ending Mortgage Balance'].apply(lambda x : round(x,2))

        month = np.zeros(self.n)
        CPR = np.zeros(self.n)
        SMM = np.zeros(self.n)
        beginning_balance = np.zeros(self.n)
        monthly_payment = np.zeros(self.n)
        monthly_interest_path_by_holders  = np.zeros(self.n)
        monthly_interest_paid_to_investors = np.zeros(self.n)
        scheduled_principal_payment = np.zeros(self.n)
        prepayment = np.zeros(self.n)
        total_principal_paid = np.zeros(self.n)
        ending_mortgage_balance = np.zeros(self.n)

        initial_monthly_payment = self.monthly_payment

        average_life = 0.0

        for i in range(self.n):

            month[i] = i + 1

            if(i + self.seasoning <= self.psa_parameters.loc[1, 'Time']):
                CPR[i] = self.psa_parameters.loc[0, 'CPR Rate'] + ((self.psa_parameters.loc[1, 'CPR Rate'] - self.psa_parameters.loc[0, 'CPR Rate']) * (
                              month[i] + self.seasoning - 1) / (self.psa_parameters.loc[1, 'Time']
                                                             - self.psa_parameters.loc[0, 'Time']))
            else:
                CPR[i] = self.psa_parameters.loc[1, 'CPR Rate'] + ((
                        self.psa_parameters.loc[2, 'CPR Rate'] - self.psa_parameters.loc[1, 'CPR Rate']) * (
                              month[i] + self.seasoning - self.psa_parameters.loc[1, 'Time']) / (
                              self.psa_parameters.loc[2, 'Time']
                              - self.psa_parameters.loc[1, 'Time']))

            CPR[i] *= self.PSA
            SMM[i] = round(1 - (1 - CPR[i])**(0.083),4)

            if i > 0:
                beginning_balance[i] = ending_mortgage_balance[i-1]
                monthly_payment[i] = (ending_mortgage_balance[i-1]*self.mortgage_rate)/(1- (1 + (self.mortgage_rate))**(-(self.n - self.seasoning - i + 1)))
                monthly_interest_paid_to_investors[i] = beginning_balance[i] * self.passthrough_rate
                monthly_interest_path_by_holders[i] = self.mortgage_rate * beginning_balance[i]
            else:
                beginning_balance[i] = self.mortgage_balance
                monthly_payment[i] = self.monthly_payment
                monthly_interest_path_by_holders[i] = self.mortgage_rate * beginning_balance[i]
                monthly_interest_paid_to_investors[i] = self.passthrough_rate * beginning_balance[i]

            scheduled_principal_payment[i] = monthly_payment[i] - monthly_interest_path_by_holders[i]
            prepayment[i] = SMM[i] * (beginning_balance[i] - scheduled_principal_payment[i])
            total_principal_paid[i] = scheduled_principal_payment[i] + prepayment[i]
            ending_mortgage_balance[i] = round(beginning_balance[i] - total_principal_paid[i],2)


        df = pd.DataFrame(data = [month,CPR,SMM,beginning_balance,monthly_payment,monthly_interest_path_by_holders,monthly_interest_paid_to_investors,scheduled_principal_payment,prepayment,
                                  total_principal_paid,ending_mortgage_balance]).transpose()
        df.columns = [['Months','Conditional Payment Rate (CPR)','Single-Monthly Mortality (SMM)','Beginning Monthly Balance',
                                                                                                      'Monthly Payment','Monthly Interest Paid in by Mortgage Holders','Monthly Interest Paid out to Investors','Scheduled Principal Payment',
                                                                                                      'Prepayment','Total Principal Payment','Ending Mortgage Balance']]
        print df.tail(10)

class SingleMortgage:
    """
    This Single Mortgage Class will serve as the Single Mortgage Cash Flow Engine
    """

    def __init__(self, **kwargs):
        """
        Use of kwargs to initialize the class as well as empty frames and lists

        :param kwargs: Keyword Arguments
        :type kwargs  :Dict
        """

        self.n = kwargs['Term']
        self.loan_balance = float(kwargs['Original_loan_balance'])
        self.rate = float(kwargs['Mortgage_rate']/12)
        self.monthly_payment = round(float((self.loan_balance * self.rate * (1 + self.rate)**self.n)/ ((1 + self.rate)**self.n - 1)),3)

        print "Your monthly payments are : %s" %str(self.monthly_payment)

        self.rge = prep_cashflow_months()
        self.prep_dataframe()


    def prep_dataframe(self):
        """
        This function will create the Dataframe containing Month, Beginning Monthly Balance, Monthly Payment, Monthly Interest, Scheduled Principal Repayment, Ending Mortgage Balance
        :return:
        """

        df = pd.DataFrame(index = pd.Series(self.rge).apply(lambda x : x.replace(day = 1)), columns = ['Monthly Payment Date','Beginning Monthly Balance','Monthly Payments','Monthly Interest',
                                                                                                      'Scheduled Principal Payment','Ending Monthly Balance'])
        df.reset_index(inplace= True)
        # df.index += 1
        df.index.name = 'Period'

        df[['Beginning Monthly Balance','Monthly Payments','Monthly Interest','Scheduled Principal Payment','Ending Monthly Balance']] = df[['Beginning Monthly Balance','Monthly Payments','Monthly Interest','Scheduled Principal Payment','Ending Monthly Balance']].astype(float)

        df['Monthly Payments'] = round(self.monthly_payment,2)
        df['Monthly Payment Date'] = df['index']
        del df['index']
        df.loc[:,'Beginning Monthly Balance'] =(1 + self.rate)**df.index*self.loan_balance  - self.monthly_payment*(((1+ self.rate)**df.index - 1)/self.rate)
        df.loc[:,'Monthly Interest'] = round(float(df.loc[:, 'Beginning Monthly Balance'] * self.rate),2)
        df.loc[:,'Scheduled Principal Payment'] = round(df.loc[:,'Monthly Payments'] - df.loc[:,'Monthly Interest'],2)
        df.loc[:,'Ending Monthly Balance'] = round(df.loc[:,'Beginning Monthly Balance'] - df.loc[:,'Scheduled Principal Payment'],2)

        """
        Just to make the Ending Monthly Balance as 0
        """

        if df.loc[359,'Beginning Monthly Balance'] < df.loc[359,'Monthly Payments'] :
             df.loc[359,'Ending Monthly Balance'] = 0

        print df.tail(5)



def main():
    """
    The main function can take hard-coded parameters that will be passed by the user
    Please pass the Annual Mortgage Rate for SingleMortgage
    :return: Nothing
    """

    config_start = datetime.now()
    # Single_Mortgage = SingleMortgage(Original_loan_balance = 100000,
    #                                  Mortgage_rate = 0.08125,
    #                                  Term = 360)

    Pass_Through = PassThroughMBS(Original_mortgage_balance = 400,
                                  mortgage_rate = .08125, pass_through_rate = .075,
                                  prepayment_multiplier = 1, Term = 360, Seasoning = 0)


if __name__ == "__main__":
    main()