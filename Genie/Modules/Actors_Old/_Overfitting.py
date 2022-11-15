"""This module contains all the overfitting functions and algorythms (source or passthrough functions)."""


# pd.set_option('display.max_columns', None)


def walkfoward_template(data, kwargs):
    """
    walkfoward_kwargs = dict(
        # <class 'function'> Function to be used to initialize the Strategy Module/script
        strategy=Strategies.MMT, # Strategies.RLGL # Strategies.STFA or # User defined function
        # Strategy must accept any column you select here
        # e.g. MMT_Strategy(open_data, low_data, high_data, close_data, parameter_data, ray_sim_n_cpus)
        columns=['open', 'low', 'high', 'close'],
        parameter_data=parameter_data,
        ray_sim_n_cpus=28,
        #
        #The following are optional, but it is how you adjust your windows for the walk forward optimization
        num_splits=10,
        n_bars=1000,
    )"""
    # Prepare Inputs and Context
    from Utils import dict_to_namedtuple
    named_kwargs = dict_to_namedtuple(kwargs, add_prefix='__')

    named_kwargs.__prices_columns = ", ".join(named_kwargs.__columns)
    nk = named_kwargs
    #
    from Utils import range_split_ohlc
    nk.__prices_columns = range_split_ohlc(data, nk.__columns, nk.__num_splits, nk.__n_bars)

    # results = nk.__strategy(*nk.__prices_columns, nk.__parameter_data, nk.__ray_sim_n_cpus)




    #fixme
    results = nk.__strategy(
        nk.__prices_columns[0],
        nk.__prices_columns[1],
        nk.__prices_columns[2],
        nk.__prices_columns[3],
        nk.__parameter_data,
        nk.__ray_sim_n_cpus
    )
    exit()

    # sr.vbt.range_split(
    #     start_idxs=pd.Index(['2020-01-01', '2020-01-02', '2020-01-01']),
    #     end_idxs=pd.Index(['2020-01-08', '2020-01-04', '2020-01-07']),
    #     plot=True)
    return results


def walkfoward_report(data, **kwargs):
    """
    Walkfoward report.
    genie_obj: Genie object or data #todo does not currently accept object
    """

    a = walkfoward_template(data, kwargs)
    # a = multiline_eval(walkfoward_template, context={'data': data, 'kwargs': kwargs})
    print(a[0][0].head())

    exit()

    return

    # return multiline_eval(walkfoward_template, context={'data': data, 'kwargs': kwargs})
