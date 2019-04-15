index = 1
plot = False
district_code = district_code_list[index]
district_df = front_flow_df[front_flow_df['district_code'] == district_code]
district_df.index = np.arange(len(district_df))
district_smoothing_data = np.zeros((len(district_df), 3))
# district_df[['flow_in', 'flow_out']].plot(x=district_df.index, title=f'{index}')
if plot:
    district_df[['dwell']].plot(x=district_df.index, title=f'{index}')
district_data = district_df[['dwell']].values

district_data[:14] = district_data[:14] +0.8
district_data[14:48] = district_data[14:48] * 0.6
# district_data[46] = district_data[46] * 2
district_data[46] = (district_data[45] + district_data[47]) /2.0

if plot:
    plt.show()
    plt.plot(district_data)
district_data[:79] = (district_data[:79] -2.2)* 0.7 + 2.1
if plot:
    plt.show()
    plt.plot(district_data)
district_smoothing_data[:,0] = district_data[:,0]
# flow_in
if plot:
    district_df[['flow_in']].plot(x=district_df.index, title=f'{index}')
district_data = district_df[['flow_in']].values

district_data[:14] = district_data[:14] 

district_data[14:48] = district_data[14:48] * 0.6 - 30
district_data[46] = (district_data[45] + district_data[47]) /2.0
if plot:
    plt.show()
    plt.plot(district_data)
district_smoothing_data[:, 1] = district_data[:,0]

# flow_out
if plot:
    district_df[['flow_out']].plot(x=district_df.index, title=f'{index}')
    district_data = district_df[['flow_out']].values

district_data[:14] = district_data[:14] 

district_data[14:48] = district_data[14:48] * 0.6 - 30
district_data[46] = (district_data[45] + district_data[47]) /2.0
if plot:
    plt.show()
    plt.plot(district_data)
district_smoothing_data[:, 2] = district_data[:,0]
# update
front_flow_df.loc[front_flow_df['district_code']==district_code, ['dwell','flow_in','flow_out']] = district_smoothing_data



index = 4
plot = False
district_code = district_code_list[index]
district_df = front_flow_df[front_flow_df['district_code'] == district_code]
district_df.index = np.arange(len(district_df))
district_smoothing_data = np.zeros((len(district_df), 3))
# district_df[['flow_in', 'flow_out']].plot(x=district_df.index, title=f'{index}')
if plot:
    district_df[['dwell']].plot(x=district_df.index, title=f'{index}')
district_data = district_df[['dwell']].values

# district_data[:14] = district_data[:14] 
district_data[14:48] = district_data[14:48] * 0.3 #- 0.4
district_data[46] = (district_data[45] + district_data[47]) /2.0

if plot:
    plt.show()
    plt.plot(district_data)
print(district_data[0]-district_data[56])
district_data[:48] = (district_data[:48] - (district_data[0]-district_data[56]))#* 0.7 + 2.1
if plot:
    plt.show()
    plt.plot(district_data)
district_smoothing_data[:,0] = district_data[:,0]
# flow_in
if plot:
    district_df[['flow_in']].plot(x=district_df.index, title=f'{index}')
district_data = district_df[['flow_in']].values

district_data[:14] = district_data[:14] 

district_data[14:48] = district_data[14:48] * 0.3
district_data[46] = (district_data[45] + district_data[47]) /2.0
district_data[:48] = (district_data[:48] - (district_data[0]-district_data[56]))#* 0.7 + 2.1

if plot:
    plt.show()
    plt.plot(district_data)
district_smoothing_data[:, 1] = district_data[:,0]

# # flow_out
if plot:
    district_df[['flow_out']].plot(x=district_df.index, title=f'{index}')
    
district_data = district_df[['flow_out']].values
district_data[:14] = district_data[:14] 

district_data[14:48] = district_data[14:48] * 0.3
district_data[46] = (district_data[45] + district_data[47]) /2.0
district_data[:48] = (district_data[:48] - (district_data[0]-district_data[56]))#* 0.7 + 2.1

if plot:
    plt.show()
    plt.plot(district_data)
    
district_smoothing_data[:, 2] = district_data[:,0]
# update
front_flow_df.loc[front_flow_df['district_code']==district_code, ['dwell','flow_in','flow_out']] = district_smoothing_data
