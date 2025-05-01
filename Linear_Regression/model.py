import numpy as np


class LinearRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
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
        n_samples, n_features = x.shape
        x = np.c_[np.ones(n_samples), x]
        self.coef = np.random.normal(scale=0.05, size=n_features + 1)

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

                y_pred = x_batch @ self.coef
                error = y_pred - y_batch
                grad = (x_batch.T @ error) / len(y_batch) + self.get_penalty_grad()

                self.coef -= self.eta0 * grad

                if np.linalg.norm(grad) < self.tol:
                    return

            if self.early_stopping:
                y_val_pred = x_val @ self.coef
                val_loss = np.mean((y_val_pred - y_val) ** 2)

                if pred_loss - val_loss < self.tol:
                    no_improve_count += 1
                    if no_improve_count >= self.n_iter_no_change:
                        return
                else:
                    pred_loss = val_loss
                    no_improve_count = 0

    def predict(self, x):
        x = np.c_[np.ones(x.shape[0]), x]
        return x @ self.coef

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
