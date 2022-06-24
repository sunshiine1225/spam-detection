{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4d8073d1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\divya\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package averaged_perceptron_tagger to\n",
      "[nltk_data]     C:\\Users\\divya\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package averaged_perceptron_tagger is already up-to-\n",
      "[nltk_data]       date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\divya\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\divya\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package omw-1.4 to\n",
      "[nltk_data]     C:\\Users\\divya\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package omw-1.4 is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>Unnamed: 2</th>\n",
       "      <th>Unnamed: 3</th>\n",
       "      <th>Unnamed: 4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     v1                                                 v2 Unnamed: 2  \\\n",
       "0   ham  Go until jurong point, crazy.. Available only ...        NaN   \n",
       "1   ham                      Ok lar... Joking wif u oni...        NaN   \n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   \n",
       "3   ham  U dun say so early hor... U c already then say...        NaN   \n",
       "4   ham  Nah I don't think he goes to usf, he lives aro...        NaN   \n",
       "\n",
       "  Unnamed: 3 Unnamed: 4  \n",
       "0        NaN        NaN  \n",
       "1        NaN        NaN  \n",
       "2        NaN        NaN  \n",
       "3        NaN        NaN  \n",
       "4        NaN        NaN  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# importing modules\n",
    "import nltk\n",
    "nltk.download('punkt')\n",
    "nltk.download('averaged_perceptron_tagger')\n",
    "nltk.download('wordnet')\n",
    "nltk.download('stopwords')\n",
    "nltk.download('omw-1.4')\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk import pos_tag\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from collections import defaultdict\n",
    "from nltk.corpus import wordnet as wn\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import model_selection, svm\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import accuracy_score\n",
    "import seaborn as sn\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "# reading the data set\n",
    "\n",
    "df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')\n",
    "\n",
    "# data exploration\n",
    "\n",
    "df.head()\n",
    " \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "41627c64",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     v1                                                 v2\n",
       "0   ham  Go until jurong point, crazy.. Available only ...\n",
       "1   ham                      Ok lar... Joking wif u oni...\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3   ham  U dun say so early hor... U c already then say...\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# data cleaning\n",
    "df=df.drop([\"Unnamed: 2\",\"Unnamed: 3\",\"Unnamed: 4\"],axis=1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e43b6b05",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  label                                               text\n",
       "0   ham  Go until jurong point, crazy.. Available only ...\n",
       "1   ham                      Ok lar... Joking wif u oni...\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3   ham  U dun say so early hor... U c already then say...\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=df.rename(columns={\"v1\":\"label\",\"v2\":\"text\"})\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "3bf03e7d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   label                                               text\n",
       "0      1  Go until jurong point, crazy.. Available only ...\n",
       "1      1                      Ok lar... Joking wif u oni...\n",
       "2      0  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3      1  U dun say so early hor... U c already then say...\n",
       "4      1  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# label encoding\n",
    "df[\"label\"]=df[\"label\"].map({\"ham\":1,\"spam\":0})\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "989c62f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       Go until jurong point crazy Available only in ...\n",
       "1                                 Ok lar Joking wif u oni\n",
       "2       Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3             U dun say so early hor U c already then say\n",
       "4       Nah I dont think he goes to usf he lives aroun...\n",
       "                              ...                        \n",
       "5567    This is the 2nd time we have tried 2 contact u...\n",
       "5568                  Will Ì b going to esplanade fr home\n",
       "5569    Pity  was in mood for that Soany other suggest...\n",
       "5570    The guy did some bitching but I acted like id ...\n",
       "5571                            Rofl Its true to its name\n",
       "Name: text, Length: 5572, dtype: object"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# --> removing the punctuations\n",
    "import string\n",
    "PUNCT_TO_REMOVE = string.punctuation\n",
    "def remove_punctuation(text):\n",
    "    \"\"\"custom function to remove the punctuation\"\"\"\n",
    "    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))\n",
    "\n",
    "df[\"text\"] = df[\"text\"].apply(lambda text: remove_punctuation(text))\n",
    "df['text']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c141e76f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       Go jurong point crazy Available bugis n great ...\n",
       "1                                 Ok lar Joking wif u oni\n",
       "2       Free entry 2 wkly comp win FA Cup final tkts 2...\n",
       "3                     U dun say early hor U c already say\n",
       "4           Nah I dont think goes usf lives around though\n",
       "                              ...                        \n",
       "5567    This 2nd time tried 2 contact u U å£750 Pound ...\n",
       "5568                     Will Ì b going esplanade fr home\n",
       "5569                          Pity mood Soany suggestions\n",
       "5570    The guy bitching I acted like id interested bu...\n",
       "5571                                   Rofl Its true name\n",
       "Name: text, Length: 5572, dtype: object"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# --> removing the stopwords\n",
    "\n",
    "STOPWORDS = set(stopwords.words('english'))\n",
    "STOPWORDS.add('subject')\n",
    "def remove_stopwords(text):\n",
    "    \"\"\"custom function to remove the stopwords\"\"\"\n",
    "    return \" \".join([word for word in str(text).split() if word not in STOPWORDS])\n",
    "\n",
    "df[\"text\"] = df[\"text\"].apply(lambda text: remove_stopwords(text))\n",
    "df['text']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "e7b56bdd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Go jurong point crazy Available bugis n great ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Ok lar Joking wif u oni</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Free entry 2 wkly comp win FA Cup final tkts 2...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>U dun say early hor U c already say</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Nah I dont think go usf life around though</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   label                                               text\n",
       "0      1  Go jurong point crazy Available bugis n great ...\n",
       "1      1                            Ok lar Joking wif u oni\n",
       "2      0  Free entry 2 wkly comp win FA Cup final tkts 2...\n",
       "3      1                U dun say early hor U c already say\n",
       "4      1         Nah I dont think go usf life around though"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# stemming -->lemmatization\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "def lemmatize_words(text):\n",
    "    return \" \".join([lemmatizer.lemmatize(word) for word in text.split()])\n",
    "df[\"text\"] = df[\"text\"].apply(lambda text: lemmatize_words(text))\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "5f30ab00",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df['text']\n",
    "y = df['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "68d9746a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ok lar Joking wif u oni\n",
      "  (0, 4597)\t0.341370051745409\n",
      "  (0, 3976)\t0.4961312078135221\n",
      "  (0, 7482)\t0.3626803734762514\n",
      "  (0, 3198)\t0.2686938625323535\n",
      "  (0, 6880)\t0.4656108178175223\n",
      "  (0, 2675)\t0.4656108178175223\n"
     ]
    }
   ],
   "source": [
    "# splitting the data set\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)\n",
    "vectorizer = TfidfVectorizer()\n",
    "print(X_train[1])\n",
    "X_train = vectorizer.fit_transform(X_train)\n",
    "X_test = vectorizer.transform(X_test)\n",
    "print(X_train[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "3efad44c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9793583127664348\n",
      "0.9882712901580826\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# model buliding\n",
    "clf = MultinomialNB()\n",
    "\n",
    "clf.fit(X_train, y_train)\n",
    "# prediction made on the training set \n",
    "y_pred1 = clf.predict(X_train)\n",
    "\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import f1_score\n",
    "\n",
    "# evaluation metrics \n",
    "trainacc = accuracy_score(y_train, y_pred1)\n",
    "trainf1 = f1_score(y_train, y_pred1)\n",
    "cm = confusion_matrix(y_train, y_pred1)\n",
    "sn.heatmap(cm, annot=True)\n",
    "print(trainacc)\n",
    "print(trainf1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "f1d721bd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.95695067264574\n",
      "0.9753340184994861\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# prediction for test set\n",
    "\n",
    "y_pred2 = clf.predict(X_test)\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import f1_score\n",
    "trainacc2 = accuracy_score(y_test, y_pred2)\n",
    "trainf2 = f1_score(y_test, y_pred2)\n",
    "\n",
    "# evaluation metrics\n",
    "\n",
    "print(trainacc2)\n",
    "print(trainf2)\n",
    "cm = confusion_matrix(y_test, y_pred2)\n",
    "sn.heatmap(cm, annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "284b5a81",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Go jurong point crazy Available bugis n great ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Ok lar Joking wif u oni</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Free entry 2 wkly comp win FA Cup final tkts 2...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>U dun say early hor U c already say</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Nah I dont think go usf life around though</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>FreeMsg Hey darling 3 week word back Id like f...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Even brother like speak They treat like aid pa...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>As per request Melle Melle Oru Minnaminunginte...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>WINNER As valued network customer selected rec...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Had mobile 11 month U R entitled Update latest...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Im gonna home soon dont want talk stuff anymor...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>SIX chance win CASH From 100 20000 pound txt C...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>URGENT You 1 week FREE membership å£100000 Pri...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>Ive searching right word thank breather I prom...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>I HAVE A DATE ON SUNDAY WITH WILL</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>XXXMobileMovieClub To use credit click WAP lin...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>Oh kim watching</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>Eh u remember 2 spell name Yes He v naughty ma...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>Fine thatåÕs way u feel ThatåÕs way gota b</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                 text  label\n",
       "0   Go jurong point crazy Available bugis n great ...      1\n",
       "1                             Ok lar Joking wif u oni      1\n",
       "2   Free entry 2 wkly comp win FA Cup final tkts 2...      0\n",
       "3                 U dun say early hor U c already say      1\n",
       "4          Nah I dont think go usf life around though      1\n",
       "5   FreeMsg Hey darling 3 week word back Id like f...      1\n",
       "6   Even brother like speak They treat like aid pa...      1\n",
       "7   As per request Melle Melle Oru Minnaminunginte...      1\n",
       "8   WINNER As valued network customer selected rec...      0\n",
       "9   Had mobile 11 month U R entitled Update latest...      0\n",
       "10  Im gonna home soon dont want talk stuff anymor...      1\n",
       "11  SIX chance win CASH From 100 20000 pound txt C...      0\n",
       "12  URGENT You 1 week FREE membership å£100000 Pri...      0\n",
       "13  Ive searching right word thank breather I prom...      1\n",
       "14                  I HAVE A DATE ON SUNDAY WITH WILL      1\n",
       "15  XXXMobileMovieClub To use credit click WAP lin...      1\n",
       "16                                    Oh kim watching      1\n",
       "17  Eh u remember 2 spell name Yes He v naughty ma...      1\n",
       "18         Fine thatåÕs way u feel ThatåÕs way gota b      1"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# predictions on a foregin dataset.\n",
    "\n",
    "test=pd.read_csv(\"test.csv\")\n",
    "test[\"text\"]=test[\"text\"].apply(lambda text: remove_punctuation(text))\n",
    "test[\"text\"]=test[\"text\"].apply(lambda text: remove_stopwords(text))\n",
    "test[\"text\"]=test[\"text\"].apply(lambda text:  lemmatize_words(text))\n",
    "x_test = test[\"text\"]\n",
    "x_test = vectorizer.transform(x_test)\n",
    "\n",
    "y_pred = clf.predict(x_test)\n",
    "test[\"label\"] = y_pred\n",
    "test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae8eed97",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
