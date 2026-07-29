from pandas import read_csv,to_datetime
from numpy import floor
import datetime

# Load csv
Queue=read_csv('queue.txt',sep='\s+')

# Make a list of usernames
namelist=[]
for i in range(len(Queue)):
    if Queue['USER'].iloc[i] in namelist:
        continue
    else:
        namelist.append(Queue['USER'].iloc[i])

# Save the maximum number of nodes used:
Queue['CORES']=Queue['NODES']*Queue['CPUS']
maxcore=Queue['CORES'].max()

# Split queue into rungs lasting longer than a day.
verylong=Queue[Queue['TIME'].str.len()>10]
long=Queue[Queue['TIME'].str.len()==10]
short=Queue[Queue['TIME'].str.len()<9]

# Sum the runtimes for each user and write to file
u = open('Cloggerlist.txt','w+')
for name in namelist:
    # Find runs from a specific user
    Namevlong=verylong[verylong['USER']==name]
    Namelong=long[long['USER']==name]
    Nameshort=short[short['USER']==name]

    totd=0
    tots=0
    totm=0
    toth=0
    # Sum times from all calculations multiplied by the number of nodes.
    for cores in range(1,maxcore+1):
        # Read the number of days from 'long' and 'very long' (over 10 days) timestrings and sum them.
        totd=totd+Namelong[Namelong['CORES']==cores]['TIME'].str[0].astype(int).sum()*cores+ \
        Namevlong[Namevlong['CORES']==cores]['TIME'].str[0:1].astype(int).sum()*cores

        # The rest are times in hh:mm:ss format.
        timev=to_datetime(Namevlong[Namevlong['CORES']==cores]['TIME'].str[3:],format='%H:%M:%S')
        timel=to_datetime(Namelong[Namelong['CORES']==cores]['TIME'].str[2:],format='%H:%M:%S')
        times=to_datetime(Nameshort[Nameshort['CORES']==cores]['TIME'],format='%H:%M:%S')

        tots=tots+(timel.dt.second.sum()+times.dt.second.sum()+timev.dt.second.sum())*cores
        totm=totm+(timel.dt.minute.sum()+times.dt.minute.sum()+timev.dt.minute.sum())*cores
        toth=toth+(timel.dt.hour.sum()+times.dt.hour.sum()+timev.dt.hour.sum())*cores

    lm=0
    lh=0
    ld=0
    # Seconds
    lm=int(floor(tots/60))      # One minute for each 60 seconds
    sp=int(tots-60*lm)          # Seconds for printing
    # Minutes
    lh=int(floor((totm+lm)/60)) # One hour for each 60 minutes
    mp=int(totm+lm-60*lh)       # Minutes for printing
    # Hours
    ld=int(floor((toth+lh)/24)) # One day for each 24 hours
    hp=int(toth+lh-24*ld)       # Hours for printing
    # Days
    tp=totd+ld                  # Days for printing
    u.write(name+'\t'+str(tp)+'-'+str(hp)+':'+str(mp)+':'+str(sp)+'\n')
u.close()
