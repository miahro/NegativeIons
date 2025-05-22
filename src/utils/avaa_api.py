# -*- coding: utf-8 -*-
"""
Author: pkolari
Date: 2.8.2024

Description: Download data and metadata from AVAA SmartSMEAR via API
"""

import pandas as pd
import requests

try:
    from StringIO import StringIO # Python 2
except ImportError:
    from io import StringIO # Python 3

BASEPATH = 'https://smear-backend.2.rahtiapp.fi'

def _toList(args):
    if not isinstance(args, list) and not isinstance(args, tuple):
        args = [args]
    return args

def _getMeta(url, queryparams, fmt):
    metadata = {}
    response = requests.get(url, params=queryparams)
    if response.status_code != 200:
        print('ERROR:')
        print(response.content)
    else:
        if fmt == 'json':
            metadata = response.json()
        else:
            metadata = pd.read_csv(StringIO(response.text))
    return metadata

def getData(fdate, ldate, tablevariables=[], 
            quality='ANY', interval=1, aggregation='NONE', timeout=60):
    """
    Function for downloading data through AVAA SMEAR API.
    Give fdate and ldate as datetime or pandas timestamp,
    tablevariables as list of table.column strings.
    API may return data in column order that is different from the tablevariable list in the API query.
    Data columns are reordered to ensure consistency with the tablevariable list. 
    """
    url = '/'.join([BASEPATH,'search/timeseries/csv'])
    queryparams = {'from': fdate.strftime('%Y-%m-%dT%H:%M:%S'),
                   'to': ldate.strftime('%Y-%m-%dT%H:%M:%S'),
                   'tablevariable': _toList(tablevariables),
                   'quality': quality,
                   'interval': interval, 'aggregation': aggregation}
    data0 = []
    data = []
    try:
        # Reading directly from url with pandas.read_csv would be faster with big datasets 
        # but this gives more informative error messages and possibility to set timeout    
        response = requests.get(url, params=queryparams, timeout=timeout)
        if response.status_code == 200:
            data0 = pd.read_csv(StringIO(response.text))
        else:
            print(response.reason)
            print(response.text.replace('\r','\n'))
    except requests.exceptions.Timeout:
        print('The request could not be completed in {} seconds.'.format(timeout))
    except pd.errors.EmptyDataError:
        print('No data.')
    if len(data0) > 0:
        # Convert date&time columns to datetime
        data0['Datetime'] = pd.to_datetime(data0[['Year','Month','Day','Hour','Minute','Second']])
        # Check if all tablevariables were returned
        hdr0 = list(data0.columns)
        hdr = [v for v in tablevariables if v in hdr0]
        missing = [v for v in tablevariables if v not in hdr0]
        if len(missing) > 0:
            print('WARNING! Temporal coverage of some table(s) is outside the given time span.')
            print('These columns will be missing:')
            for ms in missing:
                print('  ', ms)
        # Drop date & time columns, reorder data columns to match the tablevariables list
        data = data0.reindex(columns=['Datetime']+hdr)
    return data
    
def getMetadata(stations=[], tables=[], tablevariables=[], fmt='json'):
    """
    Function for downloading metadata through AVAA SMEAR API.
    Call the function with just one of these arguments: stations, tables or tablevariables.
    Give station(s) or table(s) as string or list/tuple of string(s)
    tablevariables as table.column string or list/tuple of table.column string(s).
    Format options: 'json' (dict), 'csv' (data frame)
    """
    formatstr = ''
    if fmt != 'json':
        formatstr = 'csv'
    url = '/'.join([BASEPATH,'search/variable',formatstr])
    queryparams = {}
    if len(tablevariables) > 0:
        queryparams = {'tablevariable': _toList(tablevariables)}
    elif len(tables) > 0:
        queryparams = {'table': _toList(tables)}
    elif len(stations) > 0:
        queryparams = {'station': _toList(stations)}
    metadata = {}
    if len(queryparams) > 0:
        metadata = _getMeta(url, queryparams, fmt)
    return metadata

def getEvents(tablevariables=[]):
    """
    Function for downloading data lifecycle events through AVAA SMEAR API.
    Give tablevariables as table.column string or list/tuple of several table.column strings.
    Note that the result does not contain information on which variable each event belongs to,
    in some cases it's better to retrieve the events for one variable at a time. 
    """
    fmt = 'json'
    url = '/'.join([BASEPATH,'search/event'])
    metadata = {}
    if len(tablevariables) > 0:
        queryparams = {'tablevariable': _toList(tablevariables)}
        metadata = _getMeta(url, queryparams, fmt)
    return metadata

    
