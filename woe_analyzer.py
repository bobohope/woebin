import numpy as np
import pandas as pd
from scipy import stats
import pandas.core.algorithms as algos
class WoeAnalyzer:
    def __init__(self):
        """class constructor
        """
        self.woe_df = pd.DataFrame()
        self.iv_df = pd.DataFrame()

    def calculate_woe(self, df3, ld = 0.25):
        """calculate values in woe df
        Args:
            df3(DataFrame): initial woe_df
            lambda(default 0.25): to avoid infinity
        Returns:
            DataFrame: dataframe containing woe of each category in feature
        """
        df3['EVENT_RATE']=df3['EVENT']/df3['COUNT']
        df3['NON_EVENT_RATE']=df3['NON_EVENT']/df3['COUNT']
        df3['DIST_EVENT']=df3['EVENT']/df3['EVENT'].sum()
        df3['DIST_NON_EVENT']=df3['NON_EVENT']/df3['NON_EVENT'].sum()
        df3['WOE']=np.log((df3['DIST_EVENT']+ld)/(df3['DIST_NON_EVENT']+ld))
        df3['IV']=df3['WOE']*(df3['DIST_EVENT']-df3['DIST_NON_EVENT'])
        df3['VAR_NAME']='PLACE_HOLDER'
        df3 = df3[['VAR_NAME','TYPE','MAP','MIN','MAX','COUNT','EVENT','NON_EVENT','EVENT_RATE','NON_EVENT_RATE',\
        'DIST_EVENT','DIST_NON_EVENT','WOE','IV']]
        df3 = df3.replace([np.inf, -np.inf],0).reset_index(drop=True)
        return df3

    def contiuous_bin(self, X, y, max_bin=20, force_bin = 3):
        """calculate weight of edvidence of the given dataframe
        Args:
            target(Series): pandas series of the target
            data(Series) :pandas DataFrame containing feature data
        Returns:
            woe_df(DataFrame): dataframe containing woe of each category in feature
        """
        df1 = pd.DataFrame({"X":X,"y":y})
        na = df1[df1["X"].isnull()]
        not_na = df1[df1["X"].notnull()]
        r = 0
        n = max_bin
        while np.abs(r)<1:
            # stop when average of X and average of y having monotonic relationship
            try:
                df1 = pd.DataFrame({"X":not_na["X"],"y":not_na["y"],"Bucket":pd.qcut(not_na["X"],n)})
                df2 = df1.groupby(["Bucket"], as_index=True)
                r, p = stats.spearmanr(df2["X"].mean(),df2["y"].mean())
                n = n-1
            except Exception as e:
                n = n-1
        if len(df2)==1:
            n = force_bin
            bins = algos.quantile(not_na["X"], np.linspace(0,1,n))
            if len(np.unique(bins))==2:
                bins = np.insert(bins,0,1)
                bins[1]=bins[1]-(bins[1]/2)
            df1 = pd.DataFrame({"X":not_na["X"],"y":not_na["y"],\
                "Bucket":pd.qcut(not_na["X"],np,unique(bins),include_lowest = True)})
            df2 = df1.groupby(["Bucket"], as_index=True)
        df3 = pd.DataFrame()
        df3['COUNT'] = df2['y'].count()
        df3['MIN']=df2['X'].min()
        df3['MAX']=df2['X'].max()
        df3['MAP']=df2.sum().y.index
        df3['EVENT']=df2['y'].sum()
        df3['NON_EVENT']=df3['COUNT']-df3['EVENT']

        if len(na.index)>0:
            df4=pd.DataFrame({'MIN':np.nan},index=[0])
            df4['MAX']=np.nan
            df4['MAP']=np.nan
            df4['COUNT']=na['y'].count()
            df4['EVENT']=na['y'].sum()
            df4['NON_EVENT']=df4['COUNT']-df4['EVENT']
            df3 = df3.append(df4, ignore_index=True)
        df3['TYPE']='continuous'
        df3 = self.calculate_woe(df3)
        return df3

    def categorical_bin(self, X, y, fill_na = True):
        """calculate weight of edvidence of the given dataframe
        Args:
            target(Series): pandas series of the target
            data(Series) :pandas DataFrame containing feature data
        Returns:
            woe_df(DataFrame): dataframe containing woe of each category in feature
        """
        df1 = pd.DataFrame({"X":X,"y":y})
        na = df1[df1["X"].isnull()]
        not_na = df1[df1["X"].notnull()]
        df2 = not_na.groupby(['X'], as_index=True)
        df3 = pd.DataFrame()
        df3['COUNT'] = df2.count().y
        df3['MIN']=df2.sum().y.index
        df3['MAX']=df3['MIN']
        df3['MAP']=df3['MIN']
        df3['EVENT']=df2.sum().y
        df3['NON_EVENT']=df3['COUNT']-df3['EVENT']

        if len(na.index)>0:
            df4=pd.DataFrame({'MIN':np.nan},index=[0])
            df4['MAX']=np.nan
            df4['MAP']=np.nan
            df4['COUNT']=na.count().y
            df4['EVENT']=na.sum().y
            df4['NON_EVENT']=df4['COUNT']-df4['EVENT']
            df3 = df3.append(df4, ignore_index=True)
        df3['TYPE']='categorical'
        df3 = self.calculate_woe(df3)
        return df3
    def fit(self, data, target, features = None):
        """calculate weight of edvidence of the given dataframe
        Args:
            feature(list): list of features that to calculate woe and iv
            target(Series): pandas series of the target
            data(DataFrame) :pandas DataFrame containing feature data
        Returns:
            woe_df(DataFrame): dataframe containing woe of each category in each feature
            iv_df(DataFrame): series containing iv sum of each feature
        """
        X = data if features == None else data[features]
        for i in X.columns:
            if np.issubdtype(X[i], np.number) and len(pd.Series.unique(X[i]))>2:
                woe_sub_df = self.contiuous_bin(X[i],target)
            else:
                woe_sub_df = self.categorical_bin(X[i],target)
            woe_sub_df['VAR_NAME']=i
            self.woe_df = self.woe_df.append(woe_sub_df, ignore_index = True)
        self.iv_df =self.woe_df.groupby(['VAR_NAME']).agg({'IV':'sum'}).reset_index()
        return self.woe_df, self.iv_df

    def transform(self, data, features= None, keep = False):
        """map original data to woe value
        Args:
            feature(list): list of features that to calculate woe and iv
            keep(bool): whether keeping original data or not, default false
            data(DataFrame) :pandas DataFrame containing feature data
        Returns:
            DataFrame: dataframe using woe values in feature values
        """
        if len(self.woe_df)==0:
            raise AssertionError('You must fit the data first before woe value transformation.')
        X = data if features == None else data[features]
        Xt = pd.DataFrame()
        for i in X.columns:
            woe_sub_df = self.woe_df[self.woe_df['VAR_NAME']==i][['TYPE','MAP','WOE']]
            #print(woe_sub_df)
            if woe_sub_df['TYPE'].values[0]=='continuous':
                woe_mapper = woe_sub_df[['MAP','WOE']].set_index(['MAP']).to_dict()['WOE']
                Xt[i]=X[i].apply(lambda x: next(woe_mapper[k] for k in woe_mapper if x in k))
            elif woe_sub_df['TYPE'].values[0]=='categorical':
                woe_mapper = woe_sub_df[['MAP','WOE']].set_index(['MAP']).to_dict()['WOE']
                Xt[i]=X[i].map(woe_mapper)
        if keep:
            return pd.concat([data, Xt.add_suffix('_t')], axis=1)
        else:
            return pd.concat([data.drop(columns=Xt.columns), Xt], axis=1)
