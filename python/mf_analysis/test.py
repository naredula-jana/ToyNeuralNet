from mf_tool import *
import sys

data = pd.read_csv("./AS88754644.csv")
mf_names = data.MF_NAME.values.tolist()
schema_names = data.SCHEME_NAME.values.tolist()
amount = data.AMOUNT.values.tolist()
units = data.UNITS.values.tolist()
trans_type = data.TRANSACTION_TYPE.values.tolist()
pan = data.PAN.values.tolist()
date = data.TRADE_DATE.values.tolist()
product_code = data.PRODUCT_CODE.values.tolist()
datalen = len(mf_names)
mf_db = MF()

i=1
while i<datalen:
    mf_db.add(mf_names[i],schema_names[i],pan[i],trans_type[i],amount[i],units[i],date[i],product_code[i])
    i = i+1
print("Total Records: ",i)  
mf_db.caluculate_historical_gain()

#---------------------------------------------------------
currentval = pd.read_csv("./CurrentValuationAS88754644.csv")

cols = currentval.columns
cols = cols.map(lambda x: x.replace(' ', '_') )
cols = cols.map(lambda x: x.replace('.', '') )
cols = cols.map(lambda x: x.replace('(', '') )
cols = cols.map(lambda x: x.replace(')', '') )
currentval.columns = cols
#print(currentval.columns)
scheme=currentval.Scheme.values.tolist()
unit_balance=currentval.Unit_Balance.values.tolist()
current_value=currentval.Current_ValueRs.values.tolist()
nav_date=currentval.NAV_Date.values.tolist()

i=0
datalen = len(scheme)
while i<datalen:
    if unit_balance[i]==0:
        unit_balance[i]=1
    #print(i," schemess:",scheme[i])
    mf_db.update_current_nav(scheme[i],current_value[i]/unit_balance[i],nav_date[i])
    i = i+1

mf_db.caluculate_current_gain()
#--------------------------------------------



schemalist = sorted(mf_db.current_schema_list, key=lambda k: k['rate'],reverse=False) 
count=0
gain=0
cum_rate=0
cum_cost=0
for rec in schemalist:
    if rec['cost'] < 1 :
        continue
    if rec['mf_type'] == "equity":
        continue
    count=count+1
    print("SCHEMA %2d %6s %40s gain:%5.1f rate:%2.2f cost:%4.2f " %(count, rec['pan'] ,'{:40.40}'.format(rec['schema']),rec['gain'],rec['rate'],rec['cost']))
    gain=gain+rec['gain']
    cum_cost = cum_cost + rec['cost']
    if count > 15 :
        break
print(" Total gain: ",round(gain,2), "rate: ",round(cum_rate/cum_cost,2), "cost: ",round(cum_cost,2))

translist = sorted(mf_db.current_trans_list, key=lambda k: k['rate'],reverse=False) 
total_gain=0
print("--------------------------------")
count=1
gain=0
cum_rate=0
cum_cost=0
for rec in translist:
    #print(count,rec)
    if rec['mf_type'] == "equity":
        continue
    print(" %2d %6s %40s gain:%5.1f days:%3d years:%2.1f rate:%2.2f cost:%4.2f " %(count, rec['pan'] ,'{:40.40}'.format(rec['schema']),rec['gain'],rec['days'],rec['days']/365,rec['rate'],rec['cost']))
    count=count+1
    gain=gain+rec['gain']
    cum_cost = cum_cost + rec['cost']
    cum_rate = cum_rate + rec['rate']*rec['cost']
    if count > 9 :
        break
print(" Total gain: ",round(gain,2), "rate: ",round(cum_rate/cum_cost,2), "cost: ",round(cum_cost,2))    
    
for pan in sorted(mf_db.mf_list.keys()):
    print("--------------------------------")
    
    print("CURRENT ANALYSIS : PAN:",pan," :" ,mf_db.mf_list[pan]['summary']['current_value']) 
    mf_count =1
    for mf_name in sorted(mf_db.mf_list[pan].keys()): 
        if mf_name == "summary" :
            continue
        print(mf_count,"   :",mf_name," :" ,mf_db.mf_list[pan][mf_name]['summary']['current_value'])                                                               
        mf_count=mf_count+1
        count=0
        for schema_name in sorted(mf_db.mf_list[pan][mf_name].keys()):
            if schema_name == "summary" :
                continue
            if mf_db.mf_list[pan][mf_name][schema_name]['current_value']['cost'] < 1:
                continue
            #print("%3d: %s gain:%8.2f cost:%11.2f rate:%4.2f days:%3.1f" %(count,'{:55.55}'.format(schema_name),mf_db.mf_list[pan][mf_name][schema_name]['current_value']['gain'],mf_db.mf_list[pan][mf_name][schema_name]['current_value']['cost'],mf_db.mf_list[pan][mf_name][schema_name]['current_value']['rate'],mf_db.mf_list[pan][mf_name][schema_name]['current_value']['days']))
            #print(mf_db.mf_list[pan][mf_name][schema_name]['units']," org_units: ",mf_db.mf_list[pan][mf_name][schema_name]['org_units'])
            #print(mf_db.mf_list[pan][mf_name][schema_name]['amount'])
            count=count+1
        

# display
all_schemes = True
for pan in sorted(mf_db.mf_list.keys()):
    print("--------------------------------")
    print("HISTORIC ANALYSIS: PAN: %s tot:%12.2f gain:%8.2d overallrate:%3.2d%% equit:%12.2f debt:%12.2f gainEquity:%8.2d(%3.2d%%) gainDebt:%8.2d(%3.2d%%)" %(pan,mf_db.mf_list[pan]['summary']['total'],mf_db.mf_list[pan]['summary']['gain'],mf_db.mf_list[pan]['summary']['overall_rate'],mf_db.mf_list[pan]['summary']['equity'],mf_db.mf_list[pan]['summary']['debt'],\
                                                                                                                    mf_db.mf_list[pan]['summary']['gain_equity'],mf_db.mf_list[pan]['summary']['equity_rate'], mf_db.mf_list[pan]['summary']['gain_debt'],mf_db.mf_list[pan]['summary']['debt_rate'])) 
    mf_count =1
    schema_count=1
    for mf_name in sorted(mf_db.mf_list[pan].keys()): 
        if mf_name == "summary" :
            continue
          
        print("%3d: %s tot:%12.2f  equit:%12.2f debt:%12.2f gain:%8.2d gainEquity:%8.2d(%3.2d%%) gainDebt:%8.2d(%3.2d%%)  " %(mf_count,mf_name,mf_db.mf_list[pan][mf_name]['summary']['total'],mf_db.mf_list[pan][mf_name]['summary']['equity'],mf_db.mf_list[pan][mf_name]['summary']['debt'],mf_db.mf_list[pan][mf_name]['summary']['gain'],mf_db.mf_list[pan][mf_name]['summary']['gain_equity'],\
                                                                                                                              mf_db.mf_list[pan][mf_name]['summary']['equity_rate'], mf_db.mf_list[pan][mf_name]['summary']['gain_debt'],mf_db.mf_list[pan][mf_name]['summary']['debt_rate']))
        mf_count=mf_count+1
        for schema_name in sorted(mf_db.mf_list[pan][mf_name].keys()):
            if schema_name == "summary" :
                continue
            if all_schemes==False and mf_db.mf_list[pan][mf_name][schema_name]['total_amount'] < 2:
                continue
            #if mf_db.mf_list[pan][mf_name][schema_name]['mf_type'] == "equity":
            #    continue
            if mf_db.mf_list[pan][mf_name][schema_name]['total_amount'] < 1:
                continue
            print("  %3d: %6s %55s cost:%12.2f gain:%6.2d rate:%3.2d%%" %(schema_count,mf_db.mf_list[pan][mf_name][schema_name]['mf_type'],'{:55.55}'.format(schema_name),mf_db.mf_list[pan][mf_name][schema_name]['total_amount'],mf_db.mf_list[pan][mf_name][schema_name]['gain'],mf_db.mf_list[pan][mf_name][schema_name]['rate']))
            schema_count=schema_count+1


