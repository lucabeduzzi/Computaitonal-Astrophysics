It is necessary to install COMPAS following the https://github.com/TeamCOMPAS/COMPAS instruction.


The output file samples.csv will contain :

{
    
    --eccentricity,  
    
    --initial-mass-1,
    
    --metallicity,
    
    --semi-major-axis,     
    
    q,                     # mass ratio between the two star (this is one of the distribution parameter)
    
    --initial-mass-2,      # q*--initial-mass-1
    
    SEED,                  # number/seed/tag of the star
    
    batch,                 # number of the batch (not relevant)
    
    final_mass_1,          # final mass of the first BH
    
    final_mass_2,          # final mass of the second BH
    
    is_hit,                # 0 not a DCO, 1 a DCO (BH-BH)
    
    mixture_weight,        # stroopwafel stuff
    
    time_common_enveloe_1, # time of the first common envelope event if it happend (n if it happend, 0 if it doesn't)
    
    time_common_enveloe_2  # time of the second common envelope event if it happend (n if it happend, 0 if it doesn't)

}


We are loking for all the BBH objects that have the flag ['is_hit'] = 1. In order to retrieve the intial radius there is a function implemented in SW that is:

from stroopwafel import utils, constants

radius_1 = utils.get_zams_radius(mass_1[index], metallicity[index])/constants.R_SOL_TO_AU   # it takes only single values of mass and metallicty, you cannot use array or list

sample.insert( "R1@zams", [21, 23, 24, 21], True)

for i in range (len(sample['--initial-mass-1'])):

    mass_1=sample['--initial-mass-1'][i]
    
    mass_2=sample['--initial-mass-2'][i]
    
    z=sample['--metallicty'][i]
    
    radius_1 = utils.get_zams_radius(mass_1[i], z[i])/constants.R_SOL_TO_AU 
    
    radius_2 = utils.get_zams_radius(mass_2[i], z[i])/constants.R_SOL_TO_AU 
    
    ecc.....
