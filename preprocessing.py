def MolGenerator(file_name, metal_num_list, seed_A): 
  fr = open('/content/perovskite_oxide.pdb', 'r') 
  lines = fr.readlines() 
  fr.close() 

  metal_name_list = ['La','Cs','Ba','Sr','Pr','Ca','Mn','Fe','Co','Ni','Cu']
  total_list = ['X']*2 + ['AA']*(34-2) + ['BB']*(94-34) + ['O']*(334-94) + ['AA']*(338-334)
  A_list = list(range(2,34))+list(range(334,338))
  A_len = len(A_list)
  B_list = list(range(34,94))
  B_len = len(B_list)
  O_list = list(range(94,334))
  O_len = len(O_list)

  for i in range(len(metal_num_list)):
    if (metal_name_list[i] in ['La','Cs','Ba','Sr','Pr','Ca']):
      lst = random.sample(A_list, round(A_len * metal_num_list[i]))
      A_list = [x for x in A_list if x not in lst]
      for num in lst:
        total_list[num] = metal_name_list[i]

    else:
      lst = random.sample(B_list, round(B_len * metal_num_list[i]))
      B_list = [x for x in B_list if x not in lst]
      for num in lst:
        total_list[num] = metal_name_list[i]
  
  fw = open('/content/'+str(seed_A).zfill(2)+'_'+file_name+'.pdb', 'w') 

  for line_num in range(len(lines)):
    if (line_num > 1) & (line_num < 338):
      atom = total_list[line_num]
      if (atom == 'O'):
        fw.write(lines[line_num])
      elif (atom in ['La','Cs','Ba','Sr','Pr','Ca']):
        fw.write(lines[line_num].replace('AA', atom)) 
      else:
        fw.write(lines[line_num].replace('BB', atom))
    elif (line_num >= 338):
      fw.write(lines[line_num])
    elif (line_num == 0):
      fw.write('COMPND new\n')
    elif (line_num == 1):
      fw.write('COMPND   1Created by VESTA\n')

  fw.close() 

"""
about XRD raw data
1. Measurements: scan axis (theta/2-theta), mode (continuous), range (absolute), start (10.0000 deg), stop (60.0000 deg), step (0.0200 deg), speed duration time (4.0), IS (2/3 deg), RS1 (13.000 mm), voltage (40 kV), current (30 mA)
2. Open .raw file in HighScore Plus 3.0e from PANalytical
3. Determine background -> automatic, granularity (2), bending factor (0), use smoothed input data
4. Smooth -> polynomial type (cubic), omit peaks, convolution range (25)
5. Get Iobs [cts]
6. Run the function: XRD_preprocessing
"""

def XRD_preprocessing(XRD_pickle):
  new_XRD_pickle = []
  for XRD in XRD_pickle:
    new_XRD = []
    XRD = XRD / np.max(XRD)
    lst_short = np.array(list(range(0, 2501*4400, 4400-1)))
    lst_long = np.array(list(range(0, 4400*2501, 2501-1)))

    for i in range(4400):
      left = lst_short[lst_short<=lst_long[i]][-1]
      right = lst_short[lst_short>=lst_long[i]][0]
      if left == right:
        new_XRD.extend(XRD[np.where(lst_short==left)])
      else:
        left_value = XRD[np.where(lst_short==left)]
        right_value = XRD[np.where(lst_short==right)]
        new_XRD.extend(((lst_long[i]-left)*right_value + (right-lst_long[i])*left_value)/(right-left))

    new_XRD_pickle.append(np.array(new_XRD))
  return np.array(new_XRD_pickle)
