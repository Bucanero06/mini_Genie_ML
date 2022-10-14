"""Create production ready module of h2o automl to be applied on tick data timeseries"""

import argparse
import logging
import sys

import h2o
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    """
    main function to run the program
    """
    h2o.init(ip=args.ip, port=args.port)
    h2o.remove_all()

    if args.load_model:
        model = h2o.load_model(args.load_model)
        uploaded_model = h2o.upload_model(model)
        logger.info("Loaded model from {}".format(args.load_model))


        exit()


    if args.ui and not args.input_file:
        input("Press anything to terminate")
        h2o.cluster().shutdown()
    else:
        # h2o.remove_all()

        # load data
        df = pd.read_csv(args.input_file)
        # df = df.drop(columns=[args.time_column])
        df = df.replace(to_replace='nan', value=np.nan)
        df = df.dropna()

        # create h2o dataframe
        h2o_df = h2o.H2OFrame(df)
        # h2o_df = h2o_df.drop(columns=['time'])

        # split data for modeling
        train, valid, test = h2o_df.split_frame(ratios=[.8, .1])
        y = args.target_column
        x = list(h2o_df.columns)
        x.remove(y)

        # run auto ml
        aml = H2OAutoML(max_runtime_secs=args.max_runtime_secs, seed=1, project_name=args.project_name, )
        aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

        # get leaderboard of models
        lb = aml.leaderboard

        # get leader model
        leader = h2o.get_model(lb[0, 'model_id'])

        # get predictions on test set
        pred = leader.predict(test)

        # save model
        # Save the best model
        h2o.save_model(model=leader, path=args.model_path + '/' + args.model_name, force=True)

        # save predictions
        h2o.export_file(pred, args.pred_path + '/' + args.pred_name, force=True)

        # save leaderboard
        h2o.export_file(lb, args.lb_path + '/' + args.lb_name, force=True)





        if args.ui:
            input("Press anything to terminate")
            h2o.cluster().shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create production ready module of h2o automl to be applied on tick data timeseries')
    parser.add_argument('-ui', type=str, default=False, help='Use UI')
    parser.add_argument('-ip', type=str, default='localhost', help='IP address of h2o server')
    parser.add_argument('-load_model', type=str, default=False, help='Load Model Path')
    parser.add_argument('-port', type=int, default=8888, help='Port of h2o server')
    parser.add_argument('-input_file', type=str, help='Path to input file')
    parser.add_argument('-time_column', type=str, default='time', help='Column Name of time')
    parser.add_argument('-target_column', type=str, default='target', help='Column Name of target')
    parser.add_argument('-max_runtime_secs', type=int, default=500, help='Maximum runtime for h2o automl')
    parser.add_argument('-project_name', type=str, default='ts_data_automl', help='Project name for h2o automl')
    parser.add_argument('-model_path', type=str, default='.', help='Path to save model to')
    parser.add_argument('-model_name', type=str, default='ts_data_automl.zip', help='Name of model to be saved')
    parser.add_argument('-pred_path', type=str, default='.', help='Path to save predictions to')
    parser.add_argument('-pred_name', type=str, default='tick_data_automl.csv', help='Name of predictions to be saved')
    parser.add_argument('-lb_path', type=str, default='.', help='Path to save leaderboard to')
    parser.add_argument('-lb_name', type=str, default='tick_data_automl_lb.csv', help='Name of leaderboard to be saved')
    parser.add_argument('-varimp_path', type=str, default='.', help='Path to save varimp to')
    parser.add_argument('-varimp_name', type=str, default='tick_data_automl_varimp.csv', help='Name of varimp to be saved')
    args = parser.parse_args()
    main(args)
    sys.exit(0)
