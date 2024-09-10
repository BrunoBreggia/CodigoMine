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
