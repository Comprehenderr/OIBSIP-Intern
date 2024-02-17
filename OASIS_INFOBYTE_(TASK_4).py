{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "### **EmailSpamDetection**"
      ],
      "metadata": {
        "id": "wmPlb1NyrLpA"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Fqsb4Fj8WxXm"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,recall_score,precision_score\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data=pd.read_csv(\"/content/drive/MyDrive/spam.csv\", encoding=\"latin1\")\n",
        "data.columns\n",
        "data.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HjKZhqaeXOph",
        "outputId": "5439cd84-6a33-470d-d8a1-1d395d1c41a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 5572 entries, 0 to 5571\n",
            "Data columns (total 5 columns):\n",
            " #   Column      Non-Null Count  Dtype \n",
            "---  ------      --------------  ----- \n",
            " 0   v1          5572 non-null   object\n",
            " 1   v2          5572 non-null   object\n",
            " 2   Unnamed: 2  50 non-null     object\n",
            " 3   Unnamed: 3  12 non-null     object\n",
            " 4   Unnamed: 4  6 non-null      object\n",
            "dtypes: object(5)\n",
            "memory usage: 217.8+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MAmmZXm7XOsZ",
        "outputId": "f94e81cc-a772-4d66-ff02-2e09d3365be5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(5572, 5)"
            ]
          },
          "metadata": {},
          "execution_count": 144
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5yL_jei5XOvJ",
        "outputId": "f6123f80-84c3-461e-d33c-073deb19b616"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "v1               0\n",
              "v2               0\n",
              "Unnamed: 2    5522\n",
              "Unnamed: 3    5560\n",
              "Unnamed: 4    5566\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 145
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 174
        },
        "id": "PRkPHNaqXOxw",
        "outputId": "7fe99cb5-bf4a-4d19-ffd8-e6c00594d590"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "          v1                      v2  \\\n",
              "count   5572                    5572   \n",
              "unique     2                    5169   \n",
              "top      ham  Sorry, I'll call later   \n",
              "freq    4825                      30   \n",
              "\n",
              "                                               Unnamed: 2  \\\n",
              "count                                                  50   \n",
              "unique                                                 43   \n",
              "top      bt not his girlfrnd... G o o d n i g h t . . .@\"   \n",
              "freq                                                    3   \n",
              "\n",
              "                   Unnamed: 3 Unnamed: 4  \n",
              "count                      12          6  \n",
              "unique                     10          5  \n",
              "top      MK17 92H. 450Ppw 16\"    GNT:-)\"  \n",
              "freq                        2          2  "
            ],
            "text/html": [
              "\n",
              "\n",
              "  <div id=\"df-60444dff-ed75-40a5-9d19-97dacdd11e61\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>count</th>\n",
              "      <td>5572</td>\n",
              "      <td>5572</td>\n",
              "      <td>50</td>\n",
              "      <td>12</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>unique</th>\n",
              "      <td>2</td>\n",
              "      <td>5169</td>\n",
              "      <td>43</td>\n",
              "      <td>10</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>top</th>\n",
              "      <td>ham</td>\n",
              "      <td>Sorry, I'll call later</td>\n",
              "      <td>bt not his girlfrnd... G o o d n i g h t . . .@\"</td>\n",
              "      <td>MK17 92H. 450Ppw 16\"</td>\n",
              "      <td>GNT:-)\"</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>freq</th>\n",
              "      <td>4825</td>\n",
              "      <td>30</td>\n",
              "      <td>3</td>\n",
              "      <td>2</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-60444dff-ed75-40a5-9d19-97dacdd11e61')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "\n",
              "\n",
              "\n",
              "    <div id=\"df-dcdf8dd5-67f6-4fe3-b12e-5d180c0a92ec\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-dcdf8dd5-67f6-4fe3-b12e-5d180c0a92ec')\"\n",
              "              title=\"Suggest charts.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "    </div>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "    <script>\n",
              "      async function quickchart(key) {\n",
              "        const containerElement = document.querySelector('#' + key);\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      }\n",
              "    </script>\n",
              "\n",
              "      <script>\n",
              "\n",
              "function displayQuickchartButton(domScope) {\n",
              "  let quickchartButtonEl =\n",
              "    domScope.querySelector('#df-dcdf8dd5-67f6-4fe3-b12e-5d180c0a92ec button.colab-df-quickchart');\n",
              "  quickchartButtonEl.style.display =\n",
              "    google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "}\n",
              "\n",
              "        displayQuickchartButton(document);\n",
              "      </script>\n",
              "      <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-60444dff-ed75-40a5-9d19-97dacdd11e61 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-60444dff-ed75-40a5-9d19-97dacdd11e61');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 146
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.drop(columns = data[['Unnamed: 2','Unnamed: 3','Unnamed: 4']], axis=1, inplace = True)\n",
        "data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "u_UX5vxTXO0R",
        "outputId": "c0190c11-4182-460f-8a69-fd633a853ea8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        v1                                                 v2\n",
              "0      ham  Go until jurong point, crazy.. Available only ...\n",
              "1      ham                      Ok lar... Joking wif u oni...\n",
              "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
              "3      ham  U dun say so early hor... U c already then say...\n",
              "4      ham  Nah I don't think he goes to usf, he lives aro...\n",
              "...    ...                                                ...\n",
              "5567  spam  This is the 2nd time we have tried 2 contact u...\n",
              "5568   ham              Will Ì_ b going to esplanade fr home?\n",
              "5569   ham  Pity, * was in mood for that. So...any other s...\n",
              "5570   ham  The guy did some bitching but I acted like i'd...\n",
              "5571   ham                         Rofl. Its true to its name\n",
              "\n",
              "[5572 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "\n",
              "  <div id=\"df-baaf0494-7d3c-4980-8cd2-53421dc337e5\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5567</th>\n",
              "      <td>spam</td>\n",
              "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5568</th>\n",
              "      <td>ham</td>\n",
              "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5569</th>\n",
              "      <td>ham</td>\n",
              "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5570</th>\n",
              "      <td>ham</td>\n",
              "      <td>The guy did some bitching but I acted like i'd...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5571</th>\n",
              "      <td>ham</td>\n",
              "      <td>Rofl. Its true to its name</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5572 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-baaf0494-7d3c-4980-8cd2-53421dc337e5')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "\n",
              "\n",
              "\n",
              "    <div id=\"df-524de9df-210c-4279-be50-63cb6fdb8993\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-524de9df-210c-4279-be50-63cb6fdb8993')\"\n",
              "              title=\"Suggest charts.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "    </div>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "    <script>\n",
              "      async function quickchart(key) {\n",
              "        const containerElement = document.querySelector('#' + key);\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      }\n",
              "    </script>\n",
              "\n",
              "      <script>\n",
              "\n",
              "function displayQuickchartButton(domScope) {\n",
              "  let quickchartButtonEl =\n",
              "    domScope.querySelector('#df-524de9df-210c-4279-be50-63cb6fdb8993 button.colab-df-quickchart');\n",
              "  quickchartButtonEl.style.display =\n",
              "    google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "}\n",
              "\n",
              "        displayQuickchartButton(document);\n",
              "      </script>\n",
              "      <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-baaf0494-7d3c-4980-8cd2-53421dc337e5 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-baaf0494-7d3c-4980-8cd2-53421dc337e5');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 147
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.columns=['spam/ham','email']\n",
        "data.loc[data['spam/ham'] == 'spam', 'spam/ham'] = 0\n",
        "data.loc[data['spam/ham'] == 'ham', 'spam/ham'] = 1\n",
        "data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "AlgEmen_XO6R",
        "outputId": "fc0a5e89-e1b4-4531-9807-7ef5c014ae43"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     spam/ham                                              email\n",
              "0           1  Go until jurong point, crazy.. Available only ...\n",
              "1           1                      Ok lar... Joking wif u oni...\n",
              "2           0  Free entry in 2 a wkly comp to win FA Cup fina...\n",
              "3           1  U dun say so early hor... U c already then say...\n",
              "4           1  Nah I don't think he goes to usf, he lives aro...\n",
              "...       ...                                                ...\n",
              "5567        0  This is the 2nd time we have tried 2 contact u...\n",
              "5568        1              Will Ì_ b going to esplanade fr home?\n",
              "5569        1  Pity, * was in mood for that. So...any other s...\n",
              "5570        1  The guy did some bitching but I acted like i'd...\n",
              "5571        1                         Rofl. Its true to its name\n",
              "\n",
              "[5572 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "\n",
              "  <div id=\"df-7a9e9aea-b8d7-4691-bd3b-178add13cb51\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>spam/ham</th>\n",
              "      <th>email</th>\n",
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
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5567</th>\n",
              "      <td>0</td>\n",
              "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5568</th>\n",
              "      <td>1</td>\n",
              "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5569</th>\n",
              "      <td>1</td>\n",
              "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5570</th>\n",
              "      <td>1</td>\n",
              "      <td>The guy did some bitching but I acted like i'd...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5571</th>\n",
              "      <td>1</td>\n",
              "      <td>Rofl. Its true to its name</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5572 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-7a9e9aea-b8d7-4691-bd3b-178add13cb51')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "\n",
              "\n",
              "\n",
              "    <div id=\"df-5c1c4fdf-7d97-46fb-891a-b87326a4b723\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-5c1c4fdf-7d97-46fb-891a-b87326a4b723')\"\n",
              "              title=\"Suggest charts.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "    </div>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "    <script>\n",
              "      async function quickchart(key) {\n",
              "        const containerElement = document.querySelector('#' + key);\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      }\n",
              "    </script>\n",
              "\n",
              "      <script>\n",
              "\n",
              "function displayQuickchartButton(domScope) {\n",
              "  let quickchartButtonEl =\n",
              "    domScope.querySelector('#df-5c1c4fdf-7d97-46fb-891a-b87326a4b723 button.colab-df-quickchart');\n",
              "  quickchartButtonEl.style.display =\n",
              "    google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "}\n",
              "\n",
              "        displayQuickchartButton(document);\n",
              "      </script>\n",
              "      <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-7a9e9aea-b8d7-4691-bd3b-178add13cb51 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-7a9e9aea-b8d7-4691-bd3b-178add13cb51');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 148
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = data.email\n",
        "y = data['spam/ham']"
      ],
      "metadata": {
        "id": "yjXH8GNGXPC6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "metadata": {
        "id": "zr1ak4UuXPFp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = 42)\n",
        "print(X.shape)\n",
        "print(xtrain.shape)\n",
        "print(xtest.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "s0Y0unu0XPIa",
        "outputId": "fd88dfba-8f55-4498-bdb7-c30e31e49279"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(5572,)\n",
            "(4457,)\n",
            "(1115,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "xtrain, xtest"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dk-AiCo6XPLK",
        "outputId": "e72befc5-94d8-4171-8240-52752672b02b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1978    No I'm in the same boat. Still here at my moms...\n",
              " 3989    (Bank of Granite issues Strong-Buy) EXPLOSIVE ...\n",
              " 3935       They r giving a second chance to rahul dengra.\n",
              " 4078       O i played smash bros  &lt;#&gt;  religiously.\n",
              " 4086    PRIVATE! Your 2003 Account Statement for 07973...\n",
              "                               ...                        \n",
              " 3772    I came hostel. I m going to sleep. Plz call me...\n",
              " 5191                               Sorry, I'll call later\n",
              " 5226        Prabha..i'm soryda..realy..frm heart i'm sory\n",
              " 5390                           Nt joking seriously i told\n",
              " 860                   In work now. Going have in few min.\n",
              " Name: email, Length: 4457, dtype: object,\n",
              " 3245    Funny fact Nobody teaches volcanoes 2 erupt, t...\n",
              " 944     I sent my scores to sophas and i had to do sec...\n",
              " 1044    We know someone who you know that fancies you....\n",
              " 2484    Only if you promise your getting out as SOON a...\n",
              " 812     Congratulations ur awarded either å£500 of CD ...\n",
              "                               ...                        \n",
              " 4264     &lt;DECIMAL&gt; m but its not a common car he...\n",
              " 2439    Rightio. 11.48 it is then. Well arent we all u...\n",
              " 5556    Yes i have. So that's why u texted. Pshew...mi...\n",
              " 4205                               Get the door, I'm here\n",
              " 4293    Kit Strip - you have been billed 150p. Netcoll...\n",
              " Name: email, Length: 1115, dtype: object)"
            ]
          },
          "metadata": {},
          "execution_count": 152
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ytrain, ytest"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W6rImgPkXPNq",
        "outputId": "3ff54127-91ad-4efc-84f6-6cbff4e1f4d2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1978    1\n",
              " 3989    0\n",
              " 3935    1\n",
              " 4078    1\n",
              " 4086    0\n",
              "        ..\n",
              " 3772    1\n",
              " 5191    1\n",
              " 5226    1\n",
              " 5390    1\n",
              " 860     1\n",
              " Name: spam/ham, Length: 4457, dtype: object,\n",
              " 3245    1\n",
              " 944     1\n",
              " 1044    0\n",
              " 2484    1\n",
              " 812     0\n",
              "        ..\n",
              " 4264    1\n",
              " 2439    1\n",
              " 5556    1\n",
              " 4205    1\n",
              " 4293    0\n",
              " Name: spam/ham, Length: 1115, dtype: object)"
            ]
          },
          "metadata": {},
          "execution_count": 153
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "feat_vect = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)\n",
        "feat_vect"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "KUlPnhPLXPQZ",
        "outputId": "9a9dd5aa-cedc-40b1-e1f2-55475a68f2a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "TfidfVectorizer(stop_words='english')"
            ],
            "text/html": [
              "<style>#sk-container-id-12 {color: black;background-color: white;}#sk-container-id-12 pre{padding: 0;}#sk-container-id-12 div.sk-toggleable {background-color: white;}#sk-container-id-12 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-12 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-12 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-12 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-12 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-12 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-12 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-12 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-12 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-12 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-12 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-12 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-12 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-12 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-12 div.sk-item {position: relative;z-index: 1;}#sk-container-id-12 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-12 div.sk-item::before, #sk-container-id-12 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-12 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-12 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-12 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-12 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-12 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-12 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-12 div.sk-label-container {text-align: center;}#sk-container-id-12 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-12 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-12\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>TfidfVectorizer(stop_words=&#x27;english&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-12\" type=\"checkbox\" checked><label for=\"sk-estimator-id-12\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">TfidfVectorizer</label><div class=\"sk-toggleable__content\"><pre>TfidfVectorizer(stop_words=&#x27;english&#x27;)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 154
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ytrain = ytrain.astype('int')\n",
        "ytest = ytest.astype('int')\n",
        "xtrain_vec = feat_vect.fit_transform(xtrain)\n",
        "xtest_vec = feat_vect.transform(xtest)\n",
        "print(xtrain)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HJX4KVi2XPTL",
        "outputId": "88952e89-fe85-4d52-9178-3c81ad0a9487"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1978    No I'm in the same boat. Still here at my moms...\n",
            "3989    (Bank of Granite issues Strong-Buy) EXPLOSIVE ...\n",
            "3935       They r giving a second chance to rahul dengra.\n",
            "4078       O i played smash bros  &lt;#&gt;  religiously.\n",
            "4086    PRIVATE! Your 2003 Account Statement for 07973...\n",
            "                              ...                        \n",
            "3772    I came hostel. I m going to sleep. Plz call me...\n",
            "5191                               Sorry, I'll call later\n",
            "5226        Prabha..i'm soryda..realy..frm heart i'm sory\n",
            "5390                           Nt joking seriously i told\n",
            "860                   In work now. Going have in few min.\n",
            "Name: email, Length: 4457, dtype: object\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(xtrain_vec)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "K56RAFVRXPV5",
        "outputId": "ebd17beb-fde8-4085-aa31-02f735925c06"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  (0, 4520)\t0.4658046386365619\n",
            "  (0, 3210)\t0.348722265231364\n",
            "  (0, 7415)\t0.348722265231364\n",
            "  (0, 1706)\t0.3431839629173582\n",
            "  (0, 4416)\t0.4528381701109944\n",
            "  (0, 1371)\t0.4658046386365619\n",
            "  (1, 0)\t0.2654936554684193\n",
            "  (1, 1649)\t0.3059746053542906\n",
            "  (1, 6440)\t0.2953742837684993\n",
            "  (1, 4533)\t0.3059746053542906\n",
            "  (1, 419)\t0.28715203556385105\n",
            "  (1, 4292)\t0.2953742837684993\n",
            "  (1, 5005)\t0.1937920260229529\n",
            "  (1, 2661)\t0.3059746053542906\n",
            "  (1, 1533)\t0.2015782058421696\n",
            "  (1, 6296)\t0.269833648032668\n",
            "  (1, 3631)\t0.2804339696184593\n",
            "  (1, 3140)\t0.3059746053542906\n",
            "  (1, 1187)\t0.26161139982801973\n",
            "  (2, 2190)\t0.5102109014477275\n",
            "  (2, 5351)\t0.5102109014477275\n",
            "  (2, 1674)\t0.35156722029872034\n",
            "  (2, 5770)\t0.3962151014046925\n",
            "  (2, 3061)\t0.44585171875646595\n",
            "  (3, 5484)\t0.4829129976175997\n",
            "  :\t:\n",
            "  (4451, 5740)\t0.3358090891373877\n",
            "  (4451, 4686)\t0.3478605253385091\n",
            "  (4452, 3402)\t0.4536077050510107\n",
            "  (4452, 3423)\t0.4833413012939851\n",
            "  (4452, 1579)\t0.3576443319642905\n",
            "  (4452, 1781)\t0.3311324953642251\n",
            "  (4452, 5998)\t0.3311324953642251\n",
            "  (4452, 5070)\t0.3823754823843719\n",
            "  (4452, 3085)\t0.25923599228241945\n",
            "  (4453, 6102)\t0.5894401977366341\n",
            "  (4453, 3896)\t0.6138307933697735\n",
            "  (4453, 4022)\t0.5251399912435084\n",
            "  (4454, 6108)\t0.44743083090000146\n",
            "  (4454, 2938)\t0.3775064911532845\n",
            "  (4454, 5413)\t0.40874206042479744\n",
            "  (4454, 6109)\t0.44743083090000146\n",
            "  (4454, 5161)\t0.43497582872738466\n",
            "  (4454, 3282)\t0.31753127203477105\n",
            "  (4455, 6701)\t0.40997629646174116\n",
            "  (4455, 5819)\t0.5321975225970689\n",
            "  (4455, 4674)\t0.47741741602516974\n",
            "  (4455, 3720)\t0.5663548747533713\n",
            "  (4456, 4332)\t0.6196072009518515\n",
            "  (4456, 7301)\t0.5938942231507836\n",
            "  (4456, 3085)\t0.5132022683472269\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(xtest_vec)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CFjKZJa2XPYv",
        "outputId": "dc6e55e7-aabe-4a9c-ca50-a0fc7ceaf56a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  (0, 7229)\t0.2947064107791228\n",
            "  (0, 6816)\t0.4006242977875035\n",
            "  (0, 4543)\t0.38197308370768035\n",
            "  (0, 3752)\t0.1718556592061185\n",
            "  (0, 3457)\t0.3500886226408095\n",
            "  (0, 3239)\t0.34299776014114036\n",
            "  (0, 2974)\t0.34299776014114036\n",
            "  (0, 2679)\t0.3500886226408095\n",
            "  (0, 1756)\t0.31111329907426943\n",
            "  (1, 6607)\t0.27039238853977376\n",
            "  (1, 6604)\t0.19484478334547534\n",
            "  (1, 5812)\t0.22078293973996208\n",
            "  (1, 5744)\t0.35520030142077386\n",
            "  (1, 5739)\t0.35520030142077386\n",
            "  (1, 5738)\t0.25559165628741076\n",
            "  (1, 5532)\t0.33866381848750327\n",
            "  (1, 4760)\t0.29866169283344046\n",
            "  (1, 3716)\t0.3178303138520559\n",
            "  (1, 2651)\t0.3269309971271071\n",
            "  (1, 1970)\t0.2461378627103295\n",
            "  (1, 1934)\t0.22392171769600464\n",
            "  (2, 5075)\t0.4020147546075029\n",
            "  (2, 4106)\t0.5120683436791947\n",
            "  (2, 3835)\t0.4855870501823454\n",
            "  (2, 2707)\t0.4882288103453305\n",
            "  :\t:\n",
            "  (1110, 4022)\t0.191596066847086\n",
            "  (1110, 3363)\t0.3742235014841453\n",
            "  (1110, 3174)\t0.19113154928290435\n",
            "  (1110, 2651)\t0.3742235014841453\n",
            "  (1110, 2148)\t0.301483654608874\n",
            "  (1110, 1869)\t0.3876535449194833\n",
            "  (1110, 1745)\t0.3876535449194833\n",
            "  (1110, 1599)\t0.27571108916401915\n",
            "  (1110, 1533)\t0.2553888613819584\n",
            "  (1110, 1290)\t0.2786297663784045\n",
            "  (1111, 4435)\t0.28908947793418216\n",
            "  (1111, 2451)\t0.3367130616672642\n",
            "  (1111, 1459)\t0.4550535681498989\n",
            "  (1111, 1012)\t0.4772732005778383\n",
            "  (1111, 503)\t0.4772732005778383\n",
            "  (1111, 268)\t0.37469777419250544\n",
            "  (1112, 7404)\t0.4289846764490662\n",
            "  (1112, 6561)\t0.6933358898870733\n",
            "  (1112, 4362)\t0.5790142409011977\n",
            "  (1113, 2362)\t1.0\n",
            "  (1114, 5073)\t0.39613661729871547\n",
            "  (1114, 4580)\t0.5231241609003211\n",
            "  (1114, 1408)\t0.3874225337880653\n",
            "  (1114, 1309)\t0.5418979155552347\n",
            "  (1114, 305)\t0.35449587042389225\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lr = LogisticRegression()\n",
        "lr.fit(xtrain_vec,ytrain)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "4wctPWO9XPbS",
        "outputId": "deb1dd34-4750-426a-d54c-d753dceabe57"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LogisticRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-13 {color: black;background-color: white;}#sk-container-id-13 pre{padding: 0;}#sk-container-id-13 div.sk-toggleable {background-color: white;}#sk-container-id-13 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-13 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-13 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-13 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-13 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-13 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-13 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-13 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-13 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-13 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-13 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-13 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-13 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-13 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-13 div.sk-item {position: relative;z-index: 1;}#sk-container-id-13 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-13 div.sk-item::before, #sk-container-id-13 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-13 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-13 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-13 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-13 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-13 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-13 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-13 div.sk-label-container {text-align: center;}#sk-container-id-13 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-13 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-13\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-13\" type=\"checkbox\" checked><label for=\"sk-estimator-id-13\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 158
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred_lr = lr.predict(xtest_vec)\n",
        "pred_lr"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xxIJM8V4XPeS",
        "outputId": "c7216826-21ca-4630-a338-ab9d39c16fe6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 1, 1, ..., 1, 1, 1])"
            ]
          },
          "metadata": {},
          "execution_count": 159
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix,classification_report,accuracy_score"
      ],
      "metadata": {
        "id": "NUFC0RfKXPg5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred_lr)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WPYvc5_yXPjz",
        "outputId": "f5982234-5c85-461b-9fbb-86f651bf3ffb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9524663677130045"
            ]
          },
          "metadata": {},
          "execution_count": 161
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(classification_report(ytest,pred_lr))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6rHH1S5eXPpB",
        "outputId": "e50a9b20-ebee-4acf-af20-bad6be2bf331"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.97      0.67      0.79       150\n",
            "           1       0.95      1.00      0.97       965\n",
            "\n",
            "    accuracy                           0.95      1115\n",
            "   macro avg       0.96      0.83      0.88      1115\n",
            "weighted avg       0.95      0.95      0.95      1115\n",
            "\n"
          ]
        }
      ]
    }
  ]
}
