'''This script is a copy - paste of the example notebook, without plotting capabilites. 
Temporary, cut it down when have time
use this to take energy level of specific high cutoff pt and place into text file '''

# import modules
import pandas as pd
import numpy as np
import tempfile
import logging
from hepqpr.qallse.plotting import *
from hepqpr.qallse import *
from hepqpr.qallse.dsmaker import create_dataset

# initialise the plotting module in "notebook" mode
# set_notebook_mode()

# initialise the logging module

logging.basicConfig()
fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", datefmt='%H:%M:%S')
for handler in logging.getLogger().handlers: handler.setFormatter(fmt)
logging.getLogger('hepqpr').setLevel(logging.DEBUG)

# == DATASET CONFIG

dsmaker_config = dict(
    density=0.02, # 1%
)

# == INPUT CONFIG

# whether or not to add missing doublets to the input
add_missing = True 

# == RUN CONFIG

model_class = Qallse # model class to use
extra_config = dict() # configuration arguments overriding the defaults



tempdir = tempfile.TemporaryDirectory()
print(f'using {tempdir.name}')
metas, path = create_dataset(output_path=tempdir.name, gen_doublets=True, **dsmaker_config)
#path =  '/tmp/hpt-collapse/ds10/event000001000'

with open(path + '-meta.json') as f:
    print(f.read())

# load data
dw = DataWrapper.from_path(path)
doublets = pd.read_csv(path + '-doublets.csv')
if add_missing:
    doublets = dw.add_missing_doublets(doublets)
else:
    p, r, ms = dw.compute_score(doublets)
    print(f'got {len(doublets)}.')
    print(f'  Input precision (%): {p*100:.4f}, recall (%): {r*100:.4f}, missing: {len(ms)}')
#%%time

# instantiate qallse
for x in range (1):
    model = model_class(dw, **extra_config)

    # build the qubo
    model.build_model(doublets=doublets)
    qslice = model.to_qubo()

    
    #%%time
    # execute the qubo TODO find QUBO start
    response = model.sample_qubo_slices(Q=qslice)



    # get all output doublets
    all_doublets = model.process_sample_slices(response)
    # recreate tracks and resolve remaining conflicts
    final_tracks, final_doublets = TrackRecreaterD().process_results(all_doublets)

    # stats about the qbsolv run
    # FIXME
    #en0 = dw.compute_energy(qslice)
    #en = response.record.energy[0]

   # print(f'SAMPLE -- energy: {en:.4f}, ideal: {en0:.4f} (diff: {en-en0:.6f})')
   # occs = response.record.num_occurrences
   # print(f'          best sample occurrence: {occs[0]}/{occs.sum()}')

    # scores
    p, r, missings = dw.compute_score(final_doublets)
    print(f'SCORE  -- precision (%): {p * 100}, recall (%): {r * 100}, missing: {len(missings)}')
    trackml_score = dw.compute_trackml_score(final_tracks)
    print(f'          tracks found: {len(final_tracks)}, trackml score (%): {trackml_score * 100}')
    ## omit plotted results
    ## en0, en , precision, recall, missing, tracksfound, trackscore%
    reel, fake, miss = dw.get_score_numbers(final_doublets)
    print(f' #Fakes: {fake}, Reals {reel}, Missing {miss}' )

    qann= open("LinearBIAStest2.txt","a+")
    print(f' #Fakes: {fake}, Reals {reel}, Missing {miss}' )
    #qann.write(f' #Fakes: {fake}, Reals {reel}, Missing {miss},SAMPLE -- energy: {en:.4f}, ideal: {en0:.4f} (diff: {en-en0:.6f})\n')
    '''qann.write("%d , %d " % (dw.compute_energy(Q),response.record.energy[0] ))
    qann.write(f'{p * 100}, {r * 100}, {len(missings)}, ')
    qann.write(f'{len(final_tracks)}, {trackml_score * 100}, \n')'''
