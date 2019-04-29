import pandas as pd
from efficient_apriori import apriori

movies_user = []
tupla_name = ()


def prep100k():
    col_data = ['user_id', 'movie_id', 'rating_user', 'timestamp_user']

    data = pd.read_csv("./datasets/ml-100k/u.data",
                       names=col_data,  delimiter="\t")

    col_item_data = ['movie_id', 'movie_title', 'release date', 'video_release_date',
                     'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi',
                     'Thriller', 'War', 'Western']

    itemData = pd.read_csv("./datasets/ml-100k/u.item",  names=col_item_data, encoding="ISO-8859-1",
                           delimiter="|")
    dataset = pd.merge(data, itemData, on='movie_id', how='outer')

    return dataset


def prep1m():
    col_data = ['user_id', 'movie_id', 'rating_user', 'timestamp_user']

    data = pd.read_csv("./datasets/ml-1m/ratings.dat",
                       names=col_data,  delimiter="::", engine='python')

    col_item_data = ['movie_id', 'movie_title', 'categories']

    itemData = pd.read_csv("./datasets/ml-1m/movies.dat",  names=col_item_data, encoding="ISO-8859-1",
                           delimiter="::", engine='python')

    dataset = pd.merge(data, itemData, on='movie_id', how='outer')

    return dataset


dataset = prep1m()

for name, group in dataset.groupby(['user_id']):
    tupla_name = tuple(group['movie_title'].values)
    movies_user.append(tupla_name)


itemsets, rules = apriori(movies_user, min_support=0.27,
                          min_confidence=0.58)

rules_rhs = filter(lambda rule: len(rule.lhs) ==
                   1 and len(rule.rhs) == 1, rules)

sortedRulesResult = sorted(rules_rhs, key=lambda rule: rule.lift)

print(len(sortedRulesResult))

# resultDict = []


# def calcInverseLift(conf, lift):
#     return (1-conf)/(1-(conf/lift))


# def calcChiSquare(supp, conf, lift):
#     a = (lift-1)**2
#     b = supp*conf
#     c = (conf-supp)*(lift-conf)
#     return a*(b/c)


# for rule in sortedRulesResult:
#     resultDict.append([
#         (rule.lhs, rule.rhs),
#         rule.confidence,
#         rule.support,
#         rule.lift,
#         calcInverseLift(rule.confidence, rule.lift),
#         calcChiSquare(rule.support, rule.confidence, rule.lift)
#     ])


# finalTable = pd.DataFrame(data=resultDict, columns=[
#                           'Regra', 'Confiança', 'Suporte', 'Lift A->B', 'Lift A->!B', 'ChiSquare'])

# finalTable.sort_values(
#     by='Lift A->B', ascending=False).to_excel("ordered_by_lift-1m.xlsx")

# finalTable.sort_values(
#     by='ChiSquare', ascending=False).to_excel("ordered_by_chisquare-1m.xlsx")
