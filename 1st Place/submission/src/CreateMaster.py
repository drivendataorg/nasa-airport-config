from src.GeneralUtilities import *
from src.ExtractFeatures import (LoadRawData, CrossJoinDatesAirports, ExtractAirportconfigFeatures,
                                 ExtractRunwayArrivalFeatures, ExtractRunwayDepartureFeatures,
                                 ExtractLampFeatures, ExtractETDFeatures, ExtractCrossSystemFeatures,
                                 ExtractERAFeatures, ExtractGufiTimestampFeatures, AddTargets, Adjust)


def CreateMaster(data_path: str, airports, start_time: str, end_time: str, 
                 enlarged=False, with_targets=False) -> pd.DataFrame:
    """
    Loads all the raw tables from start_time to end_time into a dictionary 
    and sequentially extracts features for each information block merging it into
    a combined master table at the airport-timestamp level with a 15minute aggregation level
    
    :param str data_path: Parent directory where the data is stored
    :param List[str] airports: List indicating which airports to create the master table for
    :param str start_time: Timestamp to read from
    :param str end_time: Timestamp to read up to 
    :param Bool enlarged: Bool indicating whether we want an enlarged version of the master table
    :param Bool with_targets: Bool indicating whether to include the targets - only for training
    
    :return pd.Dataframe master_table: Dataframe at an airport-timestamp level with all the relevant features
    """

    # Load raw data and store it in a dictionary that maps airport + key -> pd.DataFrame
    raw_data = LoadRawData(data_path=data_path, airports=airports, start_time=start_time, end_time=end_time)

    # Create cross join of all the dates between start_time and end_time at a 15min frequency
    master_table = CrossJoinDatesAirports(airports=airports, start_time=start_time, end_time=end_time)

    # Extract features for the selected data blocks and append them to the master table
    master_table = ExtractAirportconfigFeatures(master_table, raw_data['airport_config'])
    master_table = ExtractRunwayArrivalFeatures(master_table, raw_data['arrival_runway'])
    master_table = ExtractRunwayDepartureFeatures(master_table, raw_data['departure_runway'])
    master_table = ExtractLampFeatures(master_table, raw_data['lamp'])
    master_table = ExtractETDFeatures(master_table, raw_data['etd'])
    master_table = ExtractERAFeatures(master_table, raw_data['tfm_estimated_runway_arrival_time'])
    master_table = ExtractGufiTimestampFeatures(master_table, raw_data['first_position'], 'first_position')
    master_table = ExtractGufiTimestampFeatures(master_table, raw_data['mfs_runway_arrival_time'], 'mfs_runway_arrival_time')
    master_table = ExtractGufiTimestampFeatures(master_table, raw_data['mfs_runway_departure_time'], 'mfs_runway_departure_time')
    master_table = ExtractGufiTimestampFeatures(master_table, raw_data['mfs_stand_arrival_time'], 'mfs_stand_arrival_time')
    master_table = ExtractGufiTimestampFeatures(master_table, raw_data['mfs_stand_departure_time'], 'mfs_stand_departure_time')
    
    # Adjust master table in order not to have errors in edge cases in prediction time
    master_table = Adjust(master_table)

    # In case we want the master table for training we include the targets
    if with_targets:
        master_table = AddTargets(master_table)
        
    # In case we want the enlarged version of the master table
    if enlarged:
        base_feats = ['airport', 'feat_1_cat_hourofday', 
                      'feat_1_cat_dayofweek', 'feat_1_isholiday'] + [c for c in master_table.columns if 'target' in c]
        enlarged_master = master_table[['timestamp'] + base_feats].copy()

        for airport in airports:
            current = master_table[master_table['airport'] == airport].copy()
            current.drop(columns = base_feats, inplace = True)
            current.rename(columns = {f: f'{airport}_{f}' for f in current.columns if 'feat' in f}, inplace = True)
            enlarged_master = enlarged_master.merge(current, how = 'left', on = 'timestamp')
        
        master_table = enlarged_master

    return master_table