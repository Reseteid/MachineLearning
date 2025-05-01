import numpy as np


class SoftmaxRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=100,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.coef = None
        self.intercept = None

    def get_penalty_grad(self):
        if self.penalty == "l2":
            return self.alpha * self.coef * 2
        elif self.penalty == "l1":
            return self.alpha * np.sign(self.coef)
        return 0

    def fit(self, x, y):
        if self.random_state:
            np.random.seed(self.random_state)
        x = np.c_[np.ones(x.shape[0]), x]
        classes = len(set(y))
        n_samples, n_features = x.shape
        self.coef = np.random.normal(scale=0.05, size=(n_features, classes))

        if self.early_stopping:
            val = int(self.validation_fraction * n_samples)
            x_train, x_val = x[:-val], x[-val:]
            y_train, y_val = y[:-val], y[-val:]
        else:
            x_train, y_train = x, y

        no_improve_count = 0
        pred_loss = np.inf

        for _ in range(self.max_iter):
            if self.shuffle:
                indices = np.random.permutation(len(x_train))
                x_train, y_train = x_train[indices], y_train[indices]

            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                y_pred = self.softmax(x_batch @ self.coef)
                error = y_pred - np.eye(classes)[y_batch]
                grad = (x_batch.T @ error) / len(y_batch) + self.get_penalty_grad()

                self.coef -= self.eta0 * grad

                if np.linalg.norm(grad) < self.tol:
                    return

            if self.early_stopping:
                y_val_pred = x_val @ self.coef
                val_loss = np.mean((y_val_pred - np.eye(classes)[y_val]) ** 2)

                if pred_loss - val_loss < self.tol:
                    no_improve_count += 1
                    if no_improve_count >= self.n_iter_no_change:
                        return
                else:
                    pred_loss = val_loss
                    no_improve_count = 0

    def predict_proba(self, x):
        x = np.c_[np.ones(x.shape[0]), x]
        return self.softmax(x @ self.coef)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    @staticmethod
    def softmax(z):
        """
        Calculates a softmax normalization over the last axis

        Examples:

        >>> softmax(np.array([1, 2, 3]))
        [0.09003057 0.24472847 0.66524096]

        >>> softmax(np.array([[1, 2, 3], [4, 5, 6]]))
        [[0.09003057 0.24472847 0.66524096]
         [0.03511903 0.70538451 0.25949646]]
        :param z: np.array, size: (d0, d1, ..., dn)
        :return: np.array of the same size as z
        """
        exp_z = np.exp(z - z.max(axis=z.ndim-1, keepdims=True))
        return (exp_z/exp_z.sum(axis=z.ndim-1, keepdims=True))

    @property
    def coef_(self):
        return self.coef[1:]

    @property
    def intercept_(self):
        return self.coef[0]

    @coef_.setter
    def coef_(self, value):
        self.coef = value

    @intercept_.setter
    def intercept_(self, value):
        self.intercept = value
