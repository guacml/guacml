from guacml.step_tree.base_step import BaseStep


class FeatureWhitelist(BaseStep):

    def execute_inplace(self, data):
        whitelist = self.config['pre_processing']['feature_whitelist']

        if whitelist is not None:
            target = set([self.config['run_time']['target']])
            to_be_dropped = sorted(set(data.metadata.index) - set(whitelist) - target)
            data.df.drop(to_be_dropped, axis=1, inplace=True)
            data.metadata.drop(to_be_dropped, inplace=True)
            self.logger.info('FeatureWhitelist: dropped features %s', to_be_dropped)
