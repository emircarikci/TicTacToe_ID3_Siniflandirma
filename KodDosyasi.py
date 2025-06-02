
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Veri Seti Yükleme
column_names = [
    'top-left-square', 'top-middle-square', 'top-right-square',
    'middle-left-square', 'middle-middle-square', 'middle-right-square',
    'bottom-left-square', 'bottom-middle-square', 'bottom-right-square',
    'class'
]
df = pd.read_csv('tic-tac-toe.data', names=column_names)

# 2. Etiketleme (Label Encoding)
le_dict = {}
df_encoded = df.copy()
for column in df.columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    le_dict[column] = le

# 3. Özellik ve Etiket Ayırımı
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# 4. Eğitim ve Test Verisine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Karar Ağacı Modeli Eğitimi
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# 6. Tahmin ve Değerlendirme
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Doğruluk (Accuracy):", accuracy)
print("Kesinlik (Precision):", precision)
print("Duyarlılık (Recall):", recall)
print("F1-Skoru:", f1)

# 7. Confusion Matrix Görselleştirme
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_dict['class'].classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Karışıklık Matrisi")
plt.savefig("confusion_matrix.png")
plt.show()
plt.close()

# 8. Karar Ağacı Görselleştirme
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=le_dict['class'].classes_, filled=True)
plt.title("Karar Ağacı Görselleştirmesi (Entropy)")
plt.savefig("decision_tree.png")
plt.show()
plt.close()
