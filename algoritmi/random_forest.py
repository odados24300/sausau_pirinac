import numpy as np

def mse(niz):
    if len(niz) == 0:
        return 0
    else:
        suma = 0
        for i in range(len(niz)):
              suma += np.square(niz[i] - np.mean(niz))
        return suma / len(niz)

def best_split(x_matrica, y_niz, n_features_to_try=None):
    best_mse = float('inf')
    best_index = None
    best_threshold = None

    n_features = x_matrica.shape[1]

    if n_features_to_try is None:
        features_to_check = range(n_features)        # svi
    else:
        features_to_check = np.random.choice(
            n_features, 
            size=n_features_to_try, 
            replace=False                            # bez ponavljanja!
        )

    for feature in features_to_check:    # bez range()!
        thresholds = np.unique(x_matrica[:, feature])

        for threshold in thresholds:
            levi = y_niz[x_matrica[:, feature] <= threshold]
            desni = y_niz[x_matrica[:, feature] > threshold]

            if len(levi) == 0 or len(desni) == 0:
                continue
            else:
                weighted_mse = (len(levi)/len(y_niz)) * mse(levi) + (len(desni)/len(y_niz)) * mse(desni)
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_index = feature
                    best_threshold = threshold
    return best_index, best_threshold

def bootstrap_sample(X, y):
    n = X.shape[0]
    indices = np.random.choice(n, size=n, replace=True)  # sa ponavljanjem!
    return X[indices], y[indices]

# print(mse([5, 5, 5, 5]))   
# print(mse([1, 2, 3, 4]))

# print(best_split(X, y))

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features_to_try=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_to_try = n_features_to_try
        self.root = None
    
    def _grow_tree(self, X, y, depth):
        # STOP uslovi
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return Node(value=np.mean(y)) 

        feature_index, threshold = best_split(X, y, self.n_features_to_try)
        if feature_index is None:
            return Node(value=np.mean(y))
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature_index=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y, 0)

    def _traverse(self, x, node):
        # ako je list, vrati vrednost
        if node.value is not None:
            return node.value
        
        # inače biraj granu
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
        
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])
    
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.value is not None:
            print(f"{indent}→ predikcija: {node.value:.1f}")
        else:
            print(f"{indent}[X{node.feature_index} <= {node.threshold}]")
            print(f"{indent}levo:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}desno:")
            self.print_tree(node.right, depth + 1)
    
class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []   # resetuj listu
        n_features = X.shape[1] 
        k = int(np.sqrt(n_features))

        
        for _ in range(self.n_trees):
            # 1. bootstrap uzorak
            X_sample, y_sample = bootstrap_sample(X, y)
            # 2. napravi stablo (prosledi max_depth i min_samples_split!)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features_to_try=k )
            # 3. istreniraj
            tree.fit(X_sample, y_sample)
            # 4. dodaj u listu
            self.trees.append(tree)

    def predict(self, X):
        # predikcije svih stabala
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # prosek po kolonama (po uzorcima)
        return np.mean(tree_preds, axis=0)
    

# X = np.array([
#     [50000, 60],
#     [52000, 62],
#     [48000, 58],
#     [90000, 85],
#     [95000, 88],
#     [92000, 90]
# ])
# y = np.array([2000, 2100, 1950, 3200, 3300, 3250])

# tree = DecisionTree(max_depth=10, min_samples_split=2)
# tree.fit(X, y)
# tree.print_tree()


# # Test na istim podacima
# print(tree.predict(X))

# # Test na novom redu - visok Nitrogen, treba ~3200+
# print(tree.predict(np.array([[91000, 86]])))