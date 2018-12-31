
import pandas as pd
import numpy as np 
import math
from datetime import datetime
from scipy import optimize 
from datetime import date


class MF:
    def __init__(self):
        self.mf_list = {}
        self.current_trans_list=[]
        self.current_schema_list=[]
        self.cum_trans_equity_sip =[]
        self.cum_trans_equity_nonsip =[]
        self.cum_trans_debt =[]
        self.cum_trans =[]
        self.discarded_cost =0
        self.latest_date ="11-DEC-2014"
        
        self.equity_list = {"Tata Ethical Fund":"equity","Tata Digital India Fund":"equity","Tata Banking And Financial Services":"equity","Tata India Consumer Fund Regular":"equity","SBI Blue Chip Fund":"equity","Aditya Birla Sun Life Pure Value":"equity","Tata Infrastructure Fund":"equity","Tata Resources & Energy Fund":"eq","SBI Healthcare":"equity","MidCap":"equity","Midcap":"equity","Multicap":"equity","L&T Hybrid Equity Fund" : "equity", "L&T Midcap Fund":"equity","Equity":"equity"}
        self.gain=0
        self.trans_types={}
    

    def xnpv(self,rate,cashflows):
        chron_order = sorted(cashflows, key = lambda x: x[0])
        t0 = chron_order[0][0] #t0 is the date of the first cash flow

        return sum([cf/(1+rate)**((t-t0).days/365.0) for (t,cf) in chron_order])

    def cal_xirr(self,cashflows,guess=0.1): 
        #return secant_method(0.0001,lambda r: xnpv(r,cashflows),guess)
        return (optimize.newton(lambda r: self.xnpv(r,cashflows),guess)*100)

    def update_current_nav(self,schem_arg,current_nav,nav_date):
        found = False
        for pan in self.mf_list.keys():
            if pan == "summary" :
                continue
            for mf_name in self.mf_list[pan].keys():
                if mf_name == "summary" :
                    continue
                for schema_name in self.mf_list[pan][mf_name].keys():
                    if schema_name == "summary" :
                        continue
                    if self.mf_list[pan][mf_name][schema_name]['product_code'] in schem_arg:
                        self.mf_list[pan][mf_name][schema_name]['current_value']={'gain':0,'cost':0,'nav':current_nav,'nav_date':nav_date}
                        found = True
        if found==False :
            print(" Not Found :", schem_arg)
            
            
    def calculate_current_for_schema(self,pan,mf_name,schema_name):
        amount=0
        self.gain =0
        self.cost=0
        self.rate=0
        self.days=0
        self.xirr=0
        transactions =[]
        self.cost_sip =0
        self.cost_nonsip =0
        self.rate_sip =0 
        self.rate_nonsip =0
        total_units=0
        total_value_sip =0
        total_value_nonsip =0
        total_value_debut =0
        
        for  i in range(len(self.mf_list[pan][mf_name][schema_name]['units'])):
            if self.mf_list[pan][mf_name][schema_name]['units'][i] ==0:
                continue
            if self.mf_list[pan][mf_name][schema_name]['current_value']['nav'] ==0:
                self.discarded_cost= self.discarded_cost+self.mf_list[pan][mf_name][schema_name]['total_amount']
                #print("ERROR :zero nav for :",schema_name, " amount: ",self.mf_list[pan][mf_name][schema_name]['total_amount'])
                return

            curr_amount = self.mf_list[pan][mf_name][schema_name]['units'][i]*self.mf_list[pan][mf_name][schema_name]['current_value']['nav']
            curr_amount = round(curr_amount,2)
            total_units = total_units + self.mf_list[pan][mf_name][schema_name]['units'][i]
            #print("Date :",self.mf_list[pan][mf_name][schema_name]['dates'][i])
            d1 = datetime.strptime(self.mf_list[pan][mf_name][schema_name]['dates'][i], '%d-%b-%Y')
            d2 = datetime.strptime(self.mf_list[pan][mf_name][schema_name]['current_value']['nav_date'], '%d-%b-%Y')
            days=(d2-d1).days

            cost = (self.mf_list[pan][mf_name][schema_name]['units'][i]/self.mf_list[pan][mf_name][schema_name]['org_units'][i])*self.mf_list[pan][mf_name][schema_name]['amount'][i]
            
            #print(" Transtype: ",self.mf_list[pan][mf_name][schema_name]['trans_type'][i]," amount: ",self.mf_list[pan][mf_name][schema_name]['amount'][i]," orgunits: ",self.mf_list[pan][mf_name][schema_name]['org_units'][i]," newunits: ",self.mf_list[pan][mf_name][schema_name]['units'][i])
            if cost < 1 :
                continue
            gain = round((curr_amount -cost),2)
            rate = (365/days)*(gain/cost)*100
            rate = round(rate,2)
            d1 = datetime.strptime(self.mf_list[pan][mf_name][schema_name]['dates'][i], '%d-%b-%Y')
            transactions.append((d1.date(),cost))
            
            #print(" gain: ",gain," cost: ",cost," days: ",days)
            if self.mf_list[pan][mf_name][schema_name]['mf_type']=='equity' :
                if  ("SIP" in self.mf_list[pan][mf_name][schema_name]['trans_type'][i]) or ("Systematic" in self.mf_list[pan][mf_name][schema_name]['trans_type'][i])  :
                    self.cost_sip = self.cost_sip + cost
                    self.rate_sip = self.rate_sip + (rate*cost)
                    self.cum_trans_equity_sip.append((d1.date(),cost))
                    total_value_sip = total_value_sip + (self.mf_list[pan][mf_name][schema_name]['current_value']['nav'] * self.mf_list[pan][mf_name][schema_name]['units'][i])
                else:
                    self.cost_nonsip = self.cost_nonsip + cost
                    self.rate_nonsip = self.rate_nonsip + (rate*cost)
                    self.cum_trans_equity_nonsip.append((d1.date(),cost))
                    total_value_nonsip = total_value_nonsip + (self.mf_list[pan][mf_name][schema_name]['current_value']['nav'] * self.mf_list[pan][mf_name][schema_name]['units'][i])
            else:
                self.cum_trans_debt.append((d1.date(),cost))
                total_value_debut = total_value_debut + (self.mf_list[pan][mf_name][schema_name]['current_value']['nav'] * self.mf_list[pan][mf_name][schema_name]['units'][i])
            self.cum_trans.append((d1.date(),cost))
                
            rec={'curr_amount':curr_amount,'cost':cost,'gain':gain, 'rate':rate,'days':days, 'mf_type':self.mf_list[pan][mf_name][schema_name]['mf_type'],'pan':pan,'mf':mf_name,'schema':schema_name}
            self.current_trans_list.append(rec)
            
            self.gain = self.gain + gain
            self.cost = self.cost + cost
            self.days = self.days + days*cost
            self.rate = self.rate + (rate*cost)
 
        current_value = self.mf_list[pan][mf_name][schema_name]['current_value']['nav'] * total_units
        d1 = datetime.strptime(self.latest_date, '%d-%b-%Y')
        transactions.append((d1.date(),-current_value))
        self.xirr = round(self.cal_xirr(transactions),2)
        #print("xirr: ",self.xirr," current_value:",current_value,transactions )
        self.cum_trans.append((d1.date(),-current_value))
        if   total_value_sip !=0  :
            self.cum_trans_equity_sip.append((d1.date(),-total_value_sip))
        if   total_value_nonsip !=0  :
            self.cum_trans_equity_nonsip.append((d1.date(),-total_value_nonsip))
        if   total_value_debut !=0  :
            self.cum_trans_debt.append((d1.date(),-total_value_debut))

        total_cost = self.cost
        if total_cost==0 :
            total_cost=1
        self.mf_list[pan][mf_name][schema_name]['current_value']['gain'] = round(self.gain,2)
        self.mf_list[pan][mf_name][schema_name]['current_value']['cost']= round(self.cost,2)
        self.mf_list[pan][mf_name][schema_name]['current_value']['rate']= round(self.rate/total_cost,2)
        self.mf_list[pan][mf_name][schema_name]['current_value']['days']= round(self.days/total_cost,2)
        self.mf_list[pan][mf_name][schema_name]['current_value']['xirr']= self.xirr
        
        total_nonsip_cost = self.cost_nonsip
        total_sip_cost = self.cost_sip
        if total_sip_cost==0 :
            total_sip_cost=1
        if total_nonsip_cost==0 :
            total_nonsip_cost=1
        self.mf_list[pan][mf_name][schema_name]['current_value']['cost_sip']= round(self.cost_sip,2)
        self.mf_list[pan][mf_name][schema_name]['current_value']['rate_sip']= round(self.rate_sip/total_sip_cost,2) 
        self.mf_list[pan][mf_name][schema_name]['current_value']['cost_nonsip']= round(self.cost_nonsip,2)
        self.mf_list[pan][mf_name][schema_name]['current_value']['rate_nonsip']= round(self.rate_nonsip/total_nonsip_cost,2)
        
        schema_rec=self.mf_list[pan][mf_name][schema_name]['current_value'].copy()
        schema_rec['pan']=pan
        schema_rec['mf']=mf_name
        schema_rec['schema']=schema_name
        schema_rec['mf_type']=self.mf_list[pan][mf_name][schema_name]['mf_type']
        schema_rec['xirr']=self.xirr
        self.xirr = self.xirr * self.cost
        self.current_schema_list.append(schema_rec)
                        
    def add(self,mf_name, schema_name, pan ,trans_type, amount_arg,units_arg,date_arg,product_code ):
        name = mf_name +schema_name +pan 

        if isinstance(name, str)==False:
            return
        if isinstance(product_code, str)==False:
            print("ERROR product code:",type(product_code),":",product_code)
            return
        #if mf_name != "Aditya Birla Sun Life Mutual Fund" :
        #    return
        #if schema_name != "L&T Hybrid Equity Fund Direct Plan - Growth" :
        #    return
                  
        amount=amount_arg
        units=units_arg
        if math.isnan(amount_arg):
            amount=0
        else:
            amount=round(amount_arg,2)
            
        if amount==0 :
            units=0
            
        if math.isnan(amount_arg) and math.isnan(units_arg)!=True:
            print(schema_name," UNITS trans_typ: ",trans_type," amount: ",amount_arg," units: ",units_arg)

        
        mf_type="debut"
        if  "SIP" in trans_type:
            mf_type="equity"
            self.equity_list[schema_name]="equity"
        
        for equity in self.equity_list.keys():
            if equity in schema_name:
                mf_type="equity"  
        if units != 0 :
            self.trans_types[trans_type]=mf_type
            
        rec = {'trans_type': [trans_type],'amount': [amount],'total_units':units, 'product_code':product_code,'total_amount':0, 'gain':0, 'dates':[date_arg],'units':[units], 'mf_type':mf_type, 'org_units':[units],'current_value':{'nav':0,'gain':0,'cost':0,'rate':0, 'days':0}}
       
        
        if pan in self.mf_list.keys():
            if mf_name in self.mf_list[pan].keys():
                if schema_name in self.mf_list[pan][mf_name].keys():
                    self.mf_list[pan][mf_name][schema_name]['trans_type'].append(trans_type)
                    self.mf_list[pan][mf_name][schema_name]['amount'].append(amount)
                    self.mf_list[pan][mf_name][schema_name]['org_units'].append(units)
                    self.mf_list[pan][mf_name][schema_name]['units'].append(units)
                    self.mf_list[pan][mf_name][schema_name]['dates'].append(date_arg)
                    if mf_type=="equity" :
                        self.mf_list[pan][mf_name][schema_name]['mf_type'] = "equity" 
                else:
                    self.mf_list[pan][mf_name][schema_name] = rec
            else:
                self.mf_list[pan][mf_name] = {schema_name: rec}
        else:
            self.mf_list[pan] = {mf_name: {schema_name: rec}}
            
        
    def intrest_rate(self,date_from,date_to,gain,total):
        d1str = date_from
        d2str = date_to
        d1 = datetime.strptime(d1str, '%d-%b-%Y')
        d2 = datetime.strptime(d2str, '%d-%b-%Y')
        dur=(d2-d1).days
        
        latest = datetime.strptime(self.latest_date, '%d-%b-%Y')
        if d2 > latest:
            self.latest_date = d2str

        rate=(365/dur)*(gain/total)*100
        #print("from date: ",date_from," to date: ",date_to," gain: ",gain," total: ",total," duration: ",dur," rate: ",rate)
        return rate


    def caluculate_current_gain(self):
        self.current_gain=0
        ogain=0
        ocost=0
        orate=0
        oecost=0
        oerate=0
        odcost=0
        odrate=0
        for pan in (self.mf_list.keys()):
            if pan == "summary" :
                continue
            pgain =0
            pcost =0
            prate =0
            pxirr =0 
            pequity_gain =0
            pequity_cost =0
            pequity_rate =0
            pequity_xirr =0
            pdebut_gain =0
            pdebut_cost =0
            pdebut_rate =0 
            pdebut_xirr =0 
            for mf_name in (self.mf_list[pan].keys()):
                if mf_name == "summary"  :
                    continue
                mgain =0 
                mcost =0
                mrate =0
                mdays =0
                
                mequity_gain =0
                mequity_cost =0
                mequity_rate =0
                mdebut_gain =0
                mdebut_cost =0
                mdebut_rate =0 
                for schema_name in (self.mf_list[pan][mf_name].keys()):
                    if schema_name == "summary"  :
                        continue
                    self.calculate_current_for_schema(pan,mf_name,schema_name)
                    if self.cost < 1:
                        continue
                    mgain = self.gain + mgain
                    pgain = self.gain + pgain
                    mcost = self.cost + mcost
                    pcost = self.cost + pcost
                    mrate = self.rate + mrate
                    prate = self.rate + prate
                    pxirr = self.xirr + pxirr
                    if self.mf_list[pan][mf_name][schema_name]['mf_type']=="equity" :
                        mequity_gain = self.gain + mequity_gain
                        mequity_cost = self.cost + mequity_cost
                        mequity_rate = self.rate + mequity_rate
                        pequity_gain = self.gain + pequity_gain
                        pequity_cost = self.cost + pequity_cost
                        pequity_rate = self.rate + pequity_rate
                        pequity_xirr = self.xirr + pequity_xirr
                    else:
                        mdebut_gain = self.gain + mdebut_gain
                        mdebut_cost = self.cost + mdebut_cost
                        mdebut_rate = self.rate + mdebut_rate
                        pdebut_gain = self.gain + pdebut_gain
                        pdebut_cost = self.cost + pdebut_cost
                        pdebut_rate = self.rate + pdebut_rate
                        pdebut_xirr = self.xirr + pdebut_xirr
                    
                    mdays = mdays + (self.days *self.cost)

                if mequity_cost ==0 :
                    mequity_cost=1
                if mdebut_cost ==0 :
                    mdebut_cost=1
                if mcost==0 :
                    mcost=1
                self.mf_list[pan][mf_name]['summary']['current_value'] = { 'rate': round(mrate/mcost,2),'gain': round(mgain,2),'cost':round(mcost,2),'egain':round(mequity_gain,2) , 'dgain':round(mdebut_gain,2), 'dcost':round(mdebut_cost,2), 'ecost':round(mequity_cost,2),'erate':round(mequity_rate/mequity_cost,2),'drate':round(mdebut_rate/mdebut_cost,2) }
            if pcost ==0 :
                pcost=1
                self.mf_list[pan]['summary']['current_value'] = { }
                continue
            if pequity_cost ==0 :
                pequity_cost=1
            if pdebut_cost ==0 :
                pdebut_cost=1
                    
            self.mf_list[pan]['summary']['current_value'] = { 'equity_xirr':round(pequity_xirr/pequity_cost,2),'debut_xirr':round(pdebut_xirr/pdebut_cost,2), 'xirr': round(pxirr/pcost,2),'rate': round(prate/pcost,2),'gain': round(pgain,2),'cost':round(pcost,2),'egain': round(pequity_gain,2) , 'dgain':round(pdebut_gain,2), 'dcost':round(pdebut_cost,2), 'ecost':round(pequity_cost,2),'erate':round(pequity_rate/pequity_cost,2),'drate':round(pdebut_rate/pdebut_cost,2)}
            self.current_gain = self.current_gain+ round(pgain,2)
            ogain = ogain + pgain
            ocost = ocost + pcost
            orate = orate + prate
            oecost = oecost + pequity_cost
            odcost = odcost + pdebut_cost
            oerate = oerate + pequity_rate
            odrate = odrate + pdebut_rate
            
            print("CURRENT Anlaysis PAN:",pan,self.mf_list[pan]['summary']['current_value'])
        orate = round(orate/ocost,2)
        oerate = round(oerate/oecost,2)
        odrate = round(odrate/odcost,2)
        print("CURRENT OVERALL Anlaysis cost:",round(ocost,2), " rate: ",round(orate,2)," gain: ",round(ogain,2),"equitycost: ",round(oecost,2)," debtcost: ",round(odcost,2),"equityRate: ",oerate," debtrate: ",odrate)

        self.cum_equity_sip_xirr = round(self.cal_xirr(self.cum_trans_equity_sip),2)
        self.cum_equity_nonsip_xirr = round(self.cal_xirr(self.cum_trans_equity_nonsip),2)
        self.cum_debt_xirr = round(self.cal_xirr(self.cum_trans_debt),2)
        self.cum_xirr = round(self.cal_xirr(self.cum_trans),2)
        print("Current XIRR: sip :",self.cum_equity_sip_xirr," NonSip: ",self.cum_equity_nonsip_xirr," DEBT: ",self.cum_debt_xirr, " OverallXirr: ",self.cum_xirr )
        print("Latest Date: ",self.latest_date)
        print("-------------------")
            
    def caluculate_historical_gain(self):
    #calculating post gain
        total_gain_rate=0
        total_gain=0
        total_amnt=0
        for pan in sorted(self.mf_list.keys()):
            total_amnt_pan=0
            total_gain_pan=0
            total_gain_eqt_pan=0
            total_gain_dbt_pan=0
            total_equity_amnt_pan=0
            total_debt_amnt_pan=0
            total_gain_rate_pan=0
            total_gain_rate_eqt_pan=0
            total_gain_rate_dbt_pan=0

            for mf_name in sorted(self.mf_list[pan].keys()):
                if mf_name == "summary" :
                    continue
                total_amnt_mf =0
                total_gain_mf =0
                total_gain_eqt_mf=0
                total_gain_dbt_mf=0
                total_gain_rate_mf=0
                total_gain_rate_eqt_mf=0
                total_gain_rate_dbt_mf=0
                total_equity_amnt_mf=0
                total_debt_amnt_mf=0
                for schema_name in sorted(self.mf_list[pan][mf_name].keys()):
                    new_units = self.mf_list[pan][mf_name][schema_name]['units']
                    gain =0
                    cum_rate_gain =0
                    for  i in range(len(self.mf_list[pan][mf_name][schema_name]['org_units'])):
                        if self.mf_list[pan][mf_name][schema_name]['org_units'][i]==0:
                            continue
                        if self.mf_list[pan][mf_name][schema_name]['org_units'][i] < 0:
                            sell_units = -(self.mf_list[pan][mf_name][schema_name]['org_units'][i])
                            sell_price = -((self.mf_list[pan][mf_name][schema_name]['amount'][i])/sell_units)
                            for j in  range(len(new_units)):
                                if new_units[j] < 0:
                                    new_units[j]=0
                                    continue
                                if sell_units == 0:
                                    continue
                                if new_units[j] == 0:
                                    continue
                                if new_units[j] >= sell_units:
                                    new_units[j] = new_units[j] -sell_units
                                    cost_price = (self.mf_list[pan][mf_name][schema_name]['amount'][j])/(self.mf_list[pan][mf_name][schema_name]['org_units'][j])
                                    gain_diff = (sell_units*(sell_price-cost_price))
                                    gain = gain + gain_diff

                                    cost = cost_price*sell_units
                                    sell_units =0
                                    
                                    rate=self.intrest_rate(self.mf_list[pan][mf_name][schema_name]['dates'][j],self.mf_list[pan][mf_name][schema_name]['dates'][i],gain_diff,cost)
                                    #self.mf_list[pan][mf_name][schema_name]['interest_rate'][j] = rate*gain_diff
                                    cum_rate_gain = cum_rate_gain + (rate*gain_diff)
                                    continue
                                    
                                if new_units[j] < sell_units:
                                    sell_units = sell_units - new_units[j]
                                    cost_price = (self.mf_list[pan][mf_name][schema_name]['amount'][j])/(self.mf_list[pan][mf_name][schema_name]['org_units'][j])
                                    gain_diff = (new_units[j]*(sell_price-cost_price))
                                    gain = gain + gain_diff
                                    cost = cost_price*new_units[j]
                                    new_units[j] = 0 

                                    rate=self.intrest_rate(self.mf_list[pan][mf_name][schema_name]['dates'][j],self.mf_list[pan][mf_name][schema_name]['dates'][i],gain_diff,cost)
                                    #self.mf_list[pan][mf_name][schema_name]['interest_rate'][j] = rate*gain_diff
                                    cum_rate_gain = cum_rate_gain + (rate*gain_diff)

                    amount = 0
                    for  i in range(len(self.mf_list[pan][mf_name][schema_name]['units'])):
                        if self.mf_list[pan][mf_name][schema_name]['units'][i] ==0:
                            continue
                        amount = amount + (self.mf_list[pan][mf_name][schema_name]['units'][i]/self.mf_list[pan][mf_name][schema_name]['org_units'][i])*self.mf_list[pan][mf_name][schema_name]['amount'][i]
                    
                   
                        
                    if self.mf_list[pan][mf_name][schema_name]['mf_type'] == "equity" :
                        total_equity_amnt_mf = total_equity_amnt_mf + amount
                    else:
                        total_debt_amnt_mf = total_debt_amnt_mf + amount

                    self.mf_list[pan][mf_name][schema_name]['total_amount']=amount
                    self.mf_list[pan][mf_name][schema_name]['gain']=gain
                    if gain==0 :
                        self.mf_list[pan][mf_name][schema_name]['rate']=0 
                    else:
                        self.mf_list[pan][mf_name][schema_name]['rate']=cum_rate_gain/gain

                    total_gain_mf = total_gain_mf +gain
                    total_amnt_mf = total_amnt_mf+amount
                    total_gain_rate_mf = total_gain_rate_mf + cum_rate_gain
                    if self.mf_list[pan][mf_name][schema_name]['mf_type'] =="equity" :
                        total_gain_eqt_mf = total_gain_eqt_mf +gain
                        total_gain_eqt_pan = total_gain_eqt_pan +gain
                        total_gain_rate_eqt_mf = total_gain_rate_eqt_mf + cum_rate_gain
                        total_gain_rate_eqt_pan = total_gain_rate_eqt_pan + cum_rate_gain
                    else:
                        total_gain_dbt_mf = total_gain_dbt_mf +gain
                        total_gain_dbt_pan = total_gain_dbt_pan +gain  
                        total_gain_rate_dbt_mf = total_gain_rate_dbt_mf + cum_rate_gain
                        total_gain_rate_dbt_pan = total_gain_rate_dbt_pan + cum_rate_gain


                if total_gain_eqt_mf ==0:
                    total_gain_eqt_mf=1
                if total_gain_dbt_mf ==0:
                    total_gain_dbt_mf=1
                if total_gain_mf==0:
                    total_gain_mf=1           

                self.mf_list[pan][mf_name]['summary'] = {'overall_rate':(total_gain_rate_mf)/total_gain_mf,'total': total_amnt_mf,'gain':total_gain_mf, 'equity':total_equity_amnt_mf,'debt': total_debt_amnt_mf,'equity_rate':(total_gain_rate_eqt_mf/total_gain_eqt_mf),'debt_rate':(total_gain_rate_dbt_mf/total_gain_dbt_mf),'gain_equity':total_gain_eqt_mf , 'gain_debt':total_gain_dbt_mf}
                total_amnt_pan = total_amnt_pan + total_amnt_mf
                total_gain_pan = total_gain_pan + total_gain_mf
                total_gain_rate_pan = total_gain_rate_pan + total_gain_rate_mf
                total_equity_amnt_pan = total_equity_amnt_mf + total_equity_amnt_pan
                total_debt_amnt_pan = total_debt_amnt_mf + total_debt_amnt_pan
                #print("current :",self.mf_list[pan][mf_name]['summary']['current_value'])

            total_gain_rate = total_gain_rate + total_gain_rate_pan
            total_gain = total_gain+total_gain_pan
            total_amnt = total_amnt+total_amnt_pan

            if total_gain_eqt_pan ==0:
                total_gain_eqt_pan=1
            if total_gain_dbt_pan ==0:
                total_gain_dbt_pan=1 
            if total_gain_pan==0:
                total_gain_pan=1
            

            self.mf_list[pan]['summary']= {'overall_rate':(total_gain_rate_pan)/total_gain_pan,'total': total_amnt_pan,'gain':total_gain_pan, 'equity':total_equity_amnt_pan,'debt': total_debt_amnt_pan,'equity_gain':total_gain_eqt_pan,'debt_gain':total_gain_dbt_pan,'equity_rate':(total_gain_rate_eqt_pan/total_gain_eqt_pan),'debt_rate':(total_gain_rate_dbt_pan/total_gain_dbt_pan),'gain_equity':total_gain_eqt_pan , 'gain_debt':total_gain_dbt_pan}
        print("POST Analysis: Cost: %12.2f Overall Gain: %12.2f Overall Rate:%4.2f%% " %(total_amnt, total_gain,(total_gain_rate)/total_gain))            

