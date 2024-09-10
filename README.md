# CodigoMine
En este repositorio alojo los códigos de implementación de MINE (Mutual-Information Neural Estimator),
una red neuronal para la estimación de información mutua entre dos señales aleatorias, con la finalidad
de aplicarlo a señales de origen biológico.

Hacia el año 2018, Belghazi et al. publicaron un artículo en el que proponen un estimador de información
mutua basado en redes neuronales, llamado MINE (Mutual Information Neural Estimator).
Belghazi no propone una arquitectura de red en específica, sino una función a maximizar.
De esta manera, se emplea la red neuronal como un optimizador de funciones. Dicha función
se obtiene a partir de una representación particular de la divergencia de Kullback-Leibler,
que se explica en el teorema 3.3.1. Para ver demostración remitirse al paper de Belghazi et
al.

$$D_{KL}(p||q) = \sup\limits_{T:\Omega \to \mathbb{R}} \mathbb{E}_p[T] - \log \left(\mathbb{E}_q\left[e^T\right]\right)$$

Sea $\mathcal{F}$ un subconjunto cualquiera de funciones que mapea elementos según la regla 
$\Omega \to \mathbb{R}$, y tal que se cumplan las condiciones del teorema~\ref{th:donsker}, 
se tiene entonces la cota inferior:

$$D_{KL}(p||q) \ge \sup\limits_{T\in \mathcal{F}} \mathbb{E}_p[T] - \log (\mathbb{E}_q[e^T])$$

Siendo que la información mutua entre las variables aleatorias $X$ y $Y$ se define como la 
divergencia entre las distribuciones $P_{XY}(x,y)$ y $P_X(x)P_Y(y)$, tenemos:

$$I(X;Y) \ge \sup\limits_{T\in \mathcal{F}} \mathbb{E}_ {P_{XY}}[T] - \log (\mathbb{E}_{P_X P_Y}[e^T])$$

El conjunto $\mathcal{F}$ podría ser una familia de funciones 
$T_{\theta}: \mathcal{X} \mathcal{Y} \to \mathbb{R}$ parametrizada por una red neuronal con
parámetros $\theta \in \Theta$. Bajo esta suposición, consideremos el miembro derecho de 
la inecuación~\ref{eq:info-inequality} como nuestro estimador $I_{\Theta}(X;Y)$.

$$I_{\Theta}(X;Y) \equiv \sup\limits_{\theta\in \Theta} \mathbb{E}_ {P_{XY}}[T_{\theta}] - \log (\mathbb{E}_{P_X P_Y}[e^{T_{\theta}}])$$

De esta manera, reemplazando el valor del estimador~\ref{eq:mi-estimator} en la 
expresión~\ref{eq:info-inequality}, tenemos:

$$I(X;Y) \ge I_{\Theta}(X;Y)$$

La expresión~\ref{eq:lower_bound} nos indica que el estimador $I_{\Theta}(X;Y)$ posee como cota 
superior la información mutua real $I(X;Y)$. De esta manera, mediante una red neuronal buscaremos 
maximizar $I_{\Theta}(X;Y)$, lo cual se logrará mediante exploración en el hiper-espacio de parámetros 
$\Theta$ de la red.

Con esta idea en mente, presentamos MINE en la definición~\ref{def:mine}.

Sea $\mathcal{F}=\{ T_{\theta} \}_{\theta \in \Theta}$ un conjunto de funciones parametrizadas por una red neuronal. 
MINE (Mutual Information Neural Estimator) se define como:

$$\hat{I}(X;Y)_ {n} = \sup\limits_{\theta \in \Theta} \mathbb{E}_ {P_{XY}}\left[T_\theta\right] - \log \left( \mathbb{E}_{P_{X} P_{Y}}\left[e^{T_\theta}\right] \right)$$

En donde $P$ refiere a distribuciones empíricas asociadas a $n$ muestras independientes idénticamente distribuidas.

Los dos valores esperados que figuran en la definición de MINE~\ref{def:mine} se estiman a partir 
de la media de muestras empíricas con distribución $P_{XY}$ y $P_X P_Y$, respectivamente. Las muestras 
obtenidas empíricamente se asumen que poseen una distribución $P_{XY}$. Por otro lado, obtener muestras 
con distribución $P_X P_Y$ es complicado puesto que desconocemos de entrada si $X$ y $Y$ son independientes 
o no. Para generar tales muestras procedemos a barajar las muestras $Y$ de las muestras pareadas de la 
distribución conjunta $P_{XY}$, rompiendo cualquier relación existente entre $X$ y $Y$. 

Reemplazando los valores esperados por valores medios, la expresión a maximizar se convierte en 
una cantidad calculable numéricamente a partir de un conjunto finito de muestras pareadas con 
distribución $P_{XY}$. Dicha expresión se resume en~\ref{eq:mine-esperanza-estimada}, para un conjunto 
de $b$ muestras pareadas, donde el vector $\bar{\mathbf{y}}$ es una versión permutada del vector de muestras 
original $\mathbf{y}$.

$$\frac{1}{b} \sum\limits_{i=1}^{b}T_\theta(\mathbf{x}_ i,\mathbf{y}_ i) - \log \left[ \frac{1}{b} \sum\limits_{i=1}^{b} e^{T _\theta(\mathbf{x}_i,\bar{\mathbf{y}}_i)} \right]$$

