import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

anuncio = pd.read_csv('ad.data', header = None)
anuncio[1558].unique()

X = anuncio.iloc[:,0:1558].values
y = anuncio.iloc[:, 1558].values

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
# Modelo com todos os atributos
modelo1 = GaussianNB()
modelo1.fit(X_treinamento, y_treinamento)
previsoes1 = modelo1.predict(X_teste)
accuracy_score(y_teste, previsoes1)

# Seleção de atributos
selecao = SelectKBest(chi2, k=7)
X_novo = selecao.fit_transform(X, y)

# Colunas selecionadas
colunas = selecao.get_support()

X_treinamento_novo, X_teste_novo, y_treinamento, y_teste = train_test_split(X_novo, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
# Modelo com seleção de atributos
modelo2 = GaussianNB()
modelo2.fit(X_treinamento_novo, y_treinamento)
previsoes2 = modelo2.predict(X_teste_novo)
accuracy_score(y_teste, previsoes2)


