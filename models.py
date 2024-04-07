
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
def randomForest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def SVM(X_train, y_train):
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def XGBoost(X_train, y_train):
    xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_classifier.fit(X_train, y_train)
    return xgb_classifier

def logReg(X_train,y_train):
    logReg_model = LogisticRegression(random_state=42)
    logReg_model.fit(X_train,y_train)
    return logReg_model
#
def sNN(X_train, y_train):
    tf.random.set_seed(42)
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_dim=X_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    net.compile(loss='BinaryCrossentropy', optimizer='adam',
                metrics=['accuracy'])
    net.fit(X_train, y_train, epochs=100, batch_size=20)
    return net

def dNN(X_train, y_train):
    tf.random.set_seed(42)
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_dim=X_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    net.compile(loss='BinaryCrossentropy', optimizer='adam',
                metrics=['accuracy'])
    net.fit(X_train, y_train, epochs=100, batch_size=20)
    return net