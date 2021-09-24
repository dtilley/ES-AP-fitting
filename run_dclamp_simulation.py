from cell_models import protocols
from cell_models.kernik import KernikModel


def run_ind_dclamp(ind, dc_ik1=1.0, nai=10.0, ki=130.0):
    """ Create model from individual DEAP object.
    The optimized parameters are limited to the membrane conductances/fluxes.
    There is an additional parameter: phi for leak on the dynamic clamp.
     ind[0] = phi [0:1)
     ind[1]  =  'G_K1'
     ind[2]  =  'G_Kr'
     ind[3]  =  'G_Ks'
     ind[4]  =  'G_to'
     ind[5]  =  'P_CaL'
     ind[6]  =  'G_CaT'
     ind[7]  =  'G_Na'
     ind[8]  =  'G_F'
     ind[9]  =  'K_NaCa'
     ind[10] = 'P_NaK'
     ind[11] = 'G_b_Na'
     ind[12] = 'G_b_Ca'
     ind[13] = 'G_PCa'
    """

    ap_failure = False
    
    # Create the model from the DEAP individual.
    kci = KernikModel()

   
    # Apply dynamic-clamp leak
    if (ind[0] >= 0.0 and ind[0] < 1.0):
        ik1_leak = dc_ik1 * ind[0]
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak}
    else:
        print('phi != [0:1)')
        return None

   
    # Check bounds on ind
    for i in range(1, len(ind)):
        if (ind[i] < 0.0):
            print('Individual out of range.')
            return None

    # Intialize individual to model
    kci.default_parameters['G_K1']  = ind[1]
    kci.default_parameters['G_Kr']  = ind[2]
    kci.default_parameters['G_Ks']  = ind[3] 
    kci.default_parameters['G_to']  = ind[4] 
    kci.default_parameters['P_CaL'] = ind[5]
    kci.default_parameters['G_CaT'] = ind[6]
    kci.default_parameters['G_Na']  = ind[7] 
    kci.default_parameters['G_F']   = ind[8]  
    kci.default_parameters['K_NaCa']= ind[9]
    kci.default_parameters['P_NaK'] = ind[10]
    kci.default_parameters['G_b_Na']= ind[11]
    kci.default_parameters['G_b_Ca']= ind[12]
    kci.default_parameters['G_PCa'] = ind[13]

    # Set internal monovalent ion concentrations
    kci.nai_millimolar = nai
    kci.ki_millimolar = ki

    # Create 10s paced protocol
    KERNIK_PROTOCOL = protocols.PacedProtocol(model_name="Kernik", stim_end=10000, stim_mag=2)

    try:
        # Run Ishihara IK1
        tr_ishi = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        y_ishi_final = kci.y_initial
        tr_ishi.get_last_ap()
   
        # Run ICal -0.15
        kci.y_initial = y_ishi_final
        ical_leak = ind[0] * -0.15
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak, 'I_CaL': ical_leak}
        tr_ical_decrease = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        tr_ical_decrease.get_last_ap()
   
        # Run ICal 0.7
        kci.y_initial = y_ishi_final
        ical_leak = ind[0] * 0.7
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak, 'I_CaL': ical_leak}
        tr_ical_increase = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        tr_ical_increase.get_last_ap()

        # Run IKr -0.25
        kci.y_initial = y_ishi_final
        ikr_leak = ind[0] * -0.25
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak, 'I_Kr': ikr_leak}
        tr_ikr_decrease = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        tr_ikr_decrease.get_last_ap()

        # Run IKr 0.9
        kci.y_initial = y_ishi_final
        ikr_leak = ind[0] * 0.9
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak, 'I_Kr': ikr_leak}
        tr_ikr_increase = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        tr_ikr_increase.get_last_ap()

        # Run Ito -0.9
        kci.y_initial = y_ishi_final
        ito_leak = ind[0] * -0.9
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak, 'I_To': ito_leak}
        tr_ito_decrease = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        tr_ito_decrease.get_last_ap()

        # Run Ito 1.5
        kci.y_initial = y_ishi_final
        ito_leak = ind[0] * 1.5
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak, 'I_To': ito_leak}
        tr_ito_increase = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        tr_ito_increase.get_last_ap()

        # Run IKs 10
        kci.y_initial = y_ishi_final
        iks_leak = ind[0] * 10.0
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak, 'I_Ks': iks_leak}
        tr_iks_10 = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        tr_iks_10.get_last_ap()

        # Run IKs 4
        kci.y_initial = y_ishi_final
        iks_leak = ind[0] * 4.0
        kci._CellModel__no_ion_selective = {'I_K1_Ishi': ik1_leak, 'I_Ks': iks_leak}
        tr_iks_4 = kci.generate_response(KERNIK_PROTOCOL, is_no_ion_selective=True)
        tr_iks_4.get_last_ap()

        # Return AP set
        ap_set = {'cntrl': tr_ishi.last_ap,
                  '-0.15_ical': tr_ical_decrease.last_ap,
                  '0.7_ical': tr_ical_increase.last_ap,
                  '-0.25_ikr': tr_ikr_decrease.last_ap,
                  '0.9_ikr': tr_ikr_increase.last_ap,
                  '-0.9_ito': tr_ito_decrease.last_ap,
                  '1.5_ito': tr_ito_increase.last_ap,
                  '10_iks': tr_iks_10.last_ap,
                  '4_iks': tr_iks_4.last_ap}

        # Check if APs were generated
        for i in ap_set.keys():
            if ((max(ap_set[i].t)-min(ap_set[i].t)) < 800.0):
                ap_failure = True

    except OverflowError:
        ap_failure = True
        ap_set = {}
        
    return ap_set, ap_failure
