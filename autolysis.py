{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": []
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
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "YNGGbkVAblrZ"
      },
      "outputs": [],
      "source": [
        "# import libraries\n",
        "\n",
        "from google.colab import files\n",
        "import pandas as pd\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "uploaded = files.upload()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 177
        },
        "id": "6VgSFTicbrmr",
        "outputId": "de243f8c-0a2d-431b-d05f-c9ac50cbbde4"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-b1e01b01-c35e-48ef-8b93-8a6b4a0059a3\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-b1e01b01-c35e-48ef-8b93-8a6b4a0059a3\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving book_tags.csv to book_tags.csv\n",
            "Saving books.csv to books.csv\n",
            "Saving ratings.csv to ratings.csv\n",
            "Saving toread.csv to toread.csv\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Read the data"
      ],
      "metadata": {
        "id": "byghxoQyeLQX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "ratings = pd.read_csv('ratings.csv')\n",
        "ratings.head(5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "bBg14KOHbvJj",
        "outputId": "201b25b5-1166-4c3a-aa07-6e2f66cae430"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   user_id  book_id  rating\n",
              "0        2       26       4\n",
              "1        4       35       5\n",
              "2        4       26       3\n",
              "3        4      297       4\n",
              "4        4       45       4"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-5245dc14-34ee-449a-aedc-8ce721738a55\">\n",
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
              "      <th>user_id</th>\n",
              "      <th>book_id</th>\n",
              "      <th>rating</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2</td>\n",
              "      <td>26</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>4</td>\n",
              "      <td>35</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>4</td>\n",
              "      <td>26</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>297</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>45</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-5245dc14-34ee-449a-aedc-8ce721738a55')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
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
              "          document.querySelector('#df-5245dc14-34ee-449a-aedc-8ce721738a55 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-5245dc14-34ee-449a-aedc-8ce721738a55');\n",
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
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "books = pd.read_csv('books.csv')\n",
        "books.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 473
        },
        "id": "VAp3B38_c3ES",
        "outputId": "6c0add8b-4dc1-42e1-ae40-7c19a1020e4e"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   book_id  goodreads_book_id  best_book_id  work_id  books_count        isbn  \\\n",
              "0        9                960           960  3338963          311  1416524797   \n",
              "1       26                968           968  2982101          350   307277674   \n",
              "2       35                865           865  4835472          458    61122416   \n",
              "3       37             100915        100915  4790821          474    60764899   \n",
              "4       40              19501         19501  3352398          185   143038419   \n",
              "\n",
              "         isbn13                       authors  original_publication_year  \\\n",
              "0  9.781417e+12                     Dan Brown                     2000.0   \n",
              "1  9.780307e+12                     Dan Brown                     2003.0   \n",
              "2  9.780061e+12  Paulo Coelho, Alan R. Clarke                     1988.0   \n",
              "3  9.780061e+12                    C.S. Lewis                     1950.0   \n",
              "4  9.780143e+12             Elizabeth Gilbert                     2006.0   \n",
              "\n",
              "                                      original_title  ... work_ratings_count  \\\n",
              "0                                    Angels & Demons  ...            2078754   \n",
              "1                                  The Da Vinci Code  ...            1557292   \n",
              "2                                       O Alquimista  ...            1403995   \n",
              "3               The Lion, the Witch and the Wardrobe  ...            1584884   \n",
              "4  Eat, pray, love: one woman's search for everyt...  ...            1206597   \n",
              "\n",
              "  work_text_reviews_count  ratings_1  ratings_2  ratings_3  ratings_4  \\\n",
              "0                   25112      77841     145740     458429     716569   \n",
              "1                   41560      71345     126493     340790     539277   \n",
              "2                   55781      74846     123614     289143     412180   \n",
              "3                   15186      19309      55542     262038     513366   \n",
              "4                   49714     100373     149549     310212     332191   \n",
              "\n",
              "   ratings_5                                          image_url  \\\n",
              "0     680175  https://images.gr-assets.com/books/1303390735m...   \n",
              "1     479387  https://images.gr-assets.com/books/1303252999m...   \n",
              "2     504212  https://images.gr-assets.com/books/1483412266m...   \n",
              "3     734629  https://images.gr-assets.com/books/1353029077m...   \n",
              "4     314272  https://images.gr-assets.com/books/1503066414m...   \n",
              "\n",
              "                                     small_image_url  NonEnglish  \n",
              "0  https://images.gr-assets.com/books/1303390735s...           0  \n",
              "1  https://images.gr-assets.com/books/1303252999s...           0  \n",
              "2  https://images.gr-assets.com/books/1483412266s...           0  \n",
              "3  https://images.gr-assets.com/books/1353029077s...           0  \n",
              "4  https://images.gr-assets.com/books/1503066414s...           0  \n",
              "\n",
              "[5 rows x 24 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-2c21b406-534e-45f3-be72-a202160efe8e\">\n",
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
              "      <th>book_id</th>\n",
              "      <th>goodreads_book_id</th>\n",
              "      <th>best_book_id</th>\n",
              "      <th>work_id</th>\n",
              "      <th>books_count</th>\n",
              "      <th>isbn</th>\n",
              "      <th>isbn13</th>\n",
              "      <th>authors</th>\n",
              "      <th>original_publication_year</th>\n",
              "      <th>original_title</th>\n",
              "      <th>...</th>\n",
              "      <th>work_ratings_count</th>\n",
              "      <th>work_text_reviews_count</th>\n",
              "      <th>ratings_1</th>\n",
              "      <th>ratings_2</th>\n",
              "      <th>ratings_3</th>\n",
              "      <th>ratings_4</th>\n",
              "      <th>ratings_5</th>\n",
              "      <th>image_url</th>\n",
              "      <th>small_image_url</th>\n",
              "      <th>NonEnglish</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9</td>\n",
              "      <td>960</td>\n",
              "      <td>960</td>\n",
              "      <td>3338963</td>\n",
              "      <td>311</td>\n",
              "      <td>1416524797</td>\n",
              "      <td>9.781417e+12</td>\n",
              "      <td>Dan Brown</td>\n",
              "      <td>2000.0</td>\n",
              "      <td>Angels &amp; Demons</td>\n",
              "      <td>...</td>\n",
              "      <td>2078754</td>\n",
              "      <td>25112</td>\n",
              "      <td>77841</td>\n",
              "      <td>145740</td>\n",
              "      <td>458429</td>\n",
              "      <td>716569</td>\n",
              "      <td>680175</td>\n",
              "      <td>https://images.gr-assets.com/books/1303390735m...</td>\n",
              "      <td>https://images.gr-assets.com/books/1303390735s...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>26</td>\n",
              "      <td>968</td>\n",
              "      <td>968</td>\n",
              "      <td>2982101</td>\n",
              "      <td>350</td>\n",
              "      <td>307277674</td>\n",
              "      <td>9.780307e+12</td>\n",
              "      <td>Dan Brown</td>\n",
              "      <td>2003.0</td>\n",
              "      <td>The Da Vinci Code</td>\n",
              "      <td>...</td>\n",
              "      <td>1557292</td>\n",
              "      <td>41560</td>\n",
              "      <td>71345</td>\n",
              "      <td>126493</td>\n",
              "      <td>340790</td>\n",
              "      <td>539277</td>\n",
              "      <td>479387</td>\n",
              "      <td>https://images.gr-assets.com/books/1303252999m...</td>\n",
              "      <td>https://images.gr-assets.com/books/1303252999s...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>35</td>\n",
              "      <td>865</td>\n",
              "      <td>865</td>\n",
              "      <td>4835472</td>\n",
              "      <td>458</td>\n",
              "      <td>61122416</td>\n",
              "      <td>9.780061e+12</td>\n",
              "      <td>Paulo Coelho, Alan R. Clarke</td>\n",
              "      <td>1988.0</td>\n",
              "      <td>O Alquimista</td>\n",
              "      <td>...</td>\n",
              "      <td>1403995</td>\n",
              "      <td>55781</td>\n",
              "      <td>74846</td>\n",
              "      <td>123614</td>\n",
              "      <td>289143</td>\n",
              "      <td>412180</td>\n",
              "      <td>504212</td>\n",
              "      <td>https://images.gr-assets.com/books/1483412266m...</td>\n",
              "      <td>https://images.gr-assets.com/books/1483412266s...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>37</td>\n",
              "      <td>100915</td>\n",
              "      <td>100915</td>\n",
              "      <td>4790821</td>\n",
              "      <td>474</td>\n",
              "      <td>60764899</td>\n",
              "      <td>9.780061e+12</td>\n",
              "      <td>C.S. Lewis</td>\n",
              "      <td>1950.0</td>\n",
              "      <td>The Lion, the Witch and the Wardrobe</td>\n",
              "      <td>...</td>\n",
              "      <td>1584884</td>\n",
              "      <td>15186</td>\n",
              "      <td>19309</td>\n",
              "      <td>55542</td>\n",
              "      <td>262038</td>\n",
              "      <td>513366</td>\n",
              "      <td>734629</td>\n",
              "      <td>https://images.gr-assets.com/books/1353029077m...</td>\n",
              "      <td>https://images.gr-assets.com/books/1353029077s...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>40</td>\n",
              "      <td>19501</td>\n",
              "      <td>19501</td>\n",
              "      <td>3352398</td>\n",
              "      <td>185</td>\n",
              "      <td>143038419</td>\n",
              "      <td>9.780143e+12</td>\n",
              "      <td>Elizabeth Gilbert</td>\n",
              "      <td>2006.0</td>\n",
              "      <td>Eat, pray, love: one woman's search for everyt...</td>\n",
              "      <td>...</td>\n",
              "      <td>1206597</td>\n",
              "      <td>49714</td>\n",
              "      <td>100373</td>\n",
              "      <td>149549</td>\n",
              "      <td>310212</td>\n",
              "      <td>332191</td>\n",
              "      <td>314272</td>\n",
              "      <td>https://images.gr-assets.com/books/1503066414m...</td>\n",
              "      <td>https://images.gr-assets.com/books/1503066414s...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 24 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2c21b406-534e-45f3-be72-a202160efe8e')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
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
              "          document.querySelector('#df-2c21b406-534e-45f3-be72-a202160efe8e button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2c21b406-534e-45f3-be72-a202160efe8e');\n",
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
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "tags = pd.read_csv('book_tags.csv')\n",
        "tags.head(5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "2-wC6Bu8dGuL",
        "outputId": "805d7c94-5e7a-46a3-dd3b-e2e8715b0d7f"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   goodreads_book_id  tag_id  count\n",
              "0                105   26837   1356\n",
              "1                105   26771   1029\n",
              "2                105   11743    439\n",
              "3                105    8717    352\n",
              "4                105   10007    317"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-620eb081-577c-4791-a780-6e296087734c\">\n",
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
              "      <th>goodreads_book_id</th>\n",
              "      <th>tag_id</th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>105</td>\n",
              "      <td>26837</td>\n",
              "      <td>1356</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>105</td>\n",
              "      <td>26771</td>\n",
              "      <td>1029</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>105</td>\n",
              "      <td>11743</td>\n",
              "      <td>439</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>105</td>\n",
              "      <td>8717</td>\n",
              "      <td>352</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>105</td>\n",
              "      <td>10007</td>\n",
              "      <td>317</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-620eb081-577c-4791-a780-6e296087734c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
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
              "          document.querySelector('#df-620eb081-577c-4791-a780-6e296087734c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-620eb081-577c-4791-a780-6e296087734c');\n",
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
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "to_read = pd.read_csv('toread.csv')\n",
        "to_read.head(5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "YG6pyAZZd3f6",
        "outputId": "3643a2ad-cbd4-4668-9d88-5a365004da3f"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   user_id  book_id\n",
              "0       34      483\n",
              "1       94     1167\n",
              "2       29     4475\n",
              "3      116      474\n",
              "4       94     4475"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-960abd0b-dedd-49c9-9e98-2a0a75aa54d1\">\n",
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
              "      <th>user_id</th>\n",
              "      <th>book_id</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>34</td>\n",
              "      <td>483</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>94</td>\n",
              "      <td>1167</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>29</td>\n",
              "      <td>4475</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>116</td>\n",
              "      <td>474</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>94</td>\n",
              "      <td>4475</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-960abd0b-dedd-49c9-9e98-2a0a75aa54d1')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
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
              "          document.querySelector('#df-960abd0b-dedd-49c9-9e98-2a0a75aa54d1 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-960abd0b-dedd-49c9-9e98-2a0a75aa54d1');\n",
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
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Data Preprocessing\n"
      ],
      "metadata": {
        "id": "lfADAlJTeacN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "books.columns"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LZQjIguTeGdh",
        "outputId": "1361266f-7b7f-4ce2-aa3e-5b6e9d7af5d4"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['book_id', 'goodreads_book_id', 'best_book_id', 'work_id',\n",
              "       'books_count', 'isbn', 'isbn13', 'authors', 'original_publication_year',\n",
              "       'original_title', 'title', 'language_code', 'average_rating',\n",
              "       'ratings_count', 'work_ratings_count', 'work_text_reviews_count',\n",
              "       'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',\n",
              "       'image_url', 'small_image_url', 'NonEnglish'],\n",
              "      dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "books.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tdJaKmqQeYjH",
        "outputId": "6adff600-4439-407c-f9f3-8cfd895bfa8f"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "book_id                        0\n",
              "goodreads_book_id              0\n",
              "best_book_id                   0\n",
              "work_id                        0\n",
              "books_count                    0\n",
              "isbn                          16\n",
              "isbn13                        15\n",
              "authors                        0\n",
              "original_publication_year      1\n",
              "original_title                42\n",
              "title                          0\n",
              "language_code                132\n",
              "average_rating                 0\n",
              "ratings_count                  0\n",
              "work_ratings_count             0\n",
              "work_text_reviews_count        0\n",
              "ratings_1                      0\n",
              "ratings_2                      0\n",
              "ratings_3                      0\n",
              "ratings_4                      0\n",
              "ratings_5                      0\n",
              "image_url                      0\n",
              "small_image_url                0\n",
              "NonEnglish                     0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Find book_id , goodreads_book_id for records that have null values in 'original_title' in [books.csv]\n"
      ],
      "metadata": {
        "id": "awWdYkrBJL-G"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "na_books = books[books['original_title'].isnull()].book_id.to_list()\n",
        "na_books"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yxze7Hq3G79V",
        "outputId": "0ad29ce6-1ce9-4dba-dedb-f46ef44258e2"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[299,\n",
              " 464,\n",
              " 504,\n",
              " 1357,\n",
              " 1625,\n",
              " 1697,\n",
              " 1747,\n",
              " 2256,\n",
              " 2473,\n",
              " 2638,\n",
              " 2660,\n",
              " 2912,\n",
              " 3315,\n",
              " 3591,\n",
              " 3885,\n",
              " 3931,\n",
              " 4330,\n",
              " 4346,\n",
              " 4496,\n",
              " 4558,\n",
              " 5102,\n",
              " 5711,\n",
              " 5752,\n",
              " 5919,\n",
              " 6253,\n",
              " 6390,\n",
              " 6438,\n",
              " 6949,\n",
              " 7086,\n",
              " 7248,\n",
              " 7404,\n",
              " 7478,\n",
              " 7590,\n",
              " 7723,\n",
              " 7947,\n",
              " 8182,\n",
              " 8235,\n",
              " 8661,\n",
              " 8819,\n",
              " 9017,\n",
              " 9534,\n",
              " 9932]"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "na_goodreads = books[books['original_title'].isnull()].goodreads_book_id.to_list()\n",
        "na_goodreads"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4V9_S2KXIznO",
        "outputId": "42b3ba00-c47b-429f-f3d1-4aca9f9d2f99"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[7933292,\n",
              " 1923820,\n",
              " 11422,\n",
              " 29797,\n",
              " 1846,\n",
              " 11413,\n",
              " 4499669,\n",
              " 381421,\n",
              " 7497897,\n",
              " 56405,\n",
              " 81959,\n",
              " 119760,\n",
              " 17287028,\n",
              " 214332,\n",
              " 29780253,\n",
              " 11198480,\n",
              " 22632,\n",
              " 11281852,\n",
              " 643301,\n",
              " 281512,\n",
              " 12611073,\n",
              " 13547241,\n",
              " 102958,\n",
              " 265205,\n",
              " 24709327,\n",
              " 89369,\n",
              " 92254,\n",
              " 12171769,\n",
              " 99955,\n",
              " 7778609,\n",
              " 11389341,\n",
              " 15737147,\n",
              " 8765461,\n",
              " 159418,\n",
              " 5031805,\n",
              " 12878300,\n",
              " 12475019,\n",
              " 9670094,\n",
              " 46201,\n",
              " 39916,\n",
              " 1056627,\n",
              " 16182601]"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### ii) Remove the records in [books.csv] that have null values in 'original_title'.\n",
        "\n",
        "books = books.dropna(subset=['original_title'])"
      ],
      "metadata": {
        "id": "KvlRJ0PRkhGc"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "books.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YNCukbu9elG6",
        "outputId": "ed6a907d-39f3-4332-a020-f7b1d12dfd8d"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "book_id                        0\n",
              "goodreads_book_id              0\n",
              "best_book_id                   0\n",
              "work_id                        0\n",
              "books_count                    0\n",
              "isbn                          15\n",
              "isbn13                        14\n",
              "authors                        0\n",
              "original_publication_year      0\n",
              "original_title                 0\n",
              "title                          0\n",
              "language_code                124\n",
              "average_rating                 0\n",
              "ratings_count                  0\n",
              "work_ratings_count             0\n",
              "work_text_reviews_count        0\n",
              "ratings_1                      0\n",
              "ratings_2                      0\n",
              "ratings_3                      0\n",
              "ratings_4                      0\n",
              "ratings_5                      0\n",
              "image_url                      0\n",
              "small_image_url                0\n",
              "NonEnglish                     0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "  ###  Remove the records in other files [ratings.csv],[to_read.csv],[book_tags.csv] by mapping the respective book_id/goodreads_book_id (from i)\n",
        "\n",
        "ratings_na_index = ratings[ratings['book_id'].isin(na_books)].index.to_list()\n",
        "ratings_upd = ratings.drop(ratings_na_index)"
      ],
      "metadata": {
        "id": "Xe2smNLajPjy"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "to_read_na_index = to_read[to_read['book_id'].isin(na_books)].index.to_list()\n",
        "to_read_upd = to_read.drop(to_read_na_index)"
      ],
      "metadata": {
        "id": "Mo5SA8L7N0_r"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tags_na_index = tags[tags['goodreads_book_id'].isin(na_goodreads)].index.to_list()\n",
        "tags_upd = tags.drop(tags_na_index)"
      ],
      "metadata": {
        "id": "Erv30QlTQXj9"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# all the transformed df are in name_upd format\n",
        "\n",
        "books_upd = books\n",
        "books_upd.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k65LM71vjS6h",
        "outputId": "8b41d285-dfec-4155-ffdc-0c2cf158b6e5"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(833, 24)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Q12: What is the mean value of rating of all the books in the dataset based on (average_rating) [books.csv] ? (F)\n",
        "\n",
        "books_upd['average_rating'].mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k4vgt5JKlxSZ",
        "outputId": "33618e6c-613d-4762-b7de-4ab6e7145cbb"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "4.021440576230492"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Q6: Which book (original_title) has the most number of counts of tags given by the user ie. the book with maximum user records including all tags [book_tags.csv,books.csv] ? (S)\n",
        "\n",
        "### books_upd.columns\n",
        "\n",
        "## from this table we know that goodreads book id is 865\n",
        "tags_upd.groupby(['goodreads_book_id']).agg(\"sum\").sort_values(\"count\", ascending = 0)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 455
        },
        "id": "esFgy604l9-9",
        "outputId": "7e55d959-d2c2-41c8-b0c5-becdcedb08aa"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                    tag_id   count\n",
              "goodreads_book_id                 \n",
              "865                1684837  675971\n",
              "38447              1468201  469969\n",
              "119322             1702041  348935\n",
              "234225             1699064  296763\n",
              "4069               1903905  230327\n",
              "...                    ...     ...\n",
              "313620             1673811     868\n",
              "124524             1590749     808\n",
              "716696             1857984     778\n",
              "53809              2078750     772\n",
              "609870             1442786     684\n",
              "\n",
              "[851 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-11a0d473-0add-4e1d-a59a-f550661cf692\">\n",
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
              "      <th>tag_id</th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>goodreads_book_id</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>865</th>\n",
              "      <td>1684837</td>\n",
              "      <td>675971</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>38447</th>\n",
              "      <td>1468201</td>\n",
              "      <td>469969</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>119322</th>\n",
              "      <td>1702041</td>\n",
              "      <td>348935</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>234225</th>\n",
              "      <td>1699064</td>\n",
              "      <td>296763</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4069</th>\n",
              "      <td>1903905</td>\n",
              "      <td>230327</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>313620</th>\n",
              "      <td>1673811</td>\n",
              "      <td>868</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>124524</th>\n",
              "      <td>1590749</td>\n",
              "      <td>808</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>716696</th>\n",
              "      <td>1857984</td>\n",
              "      <td>778</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>53809</th>\n",
              "      <td>2078750</td>\n",
              "      <td>772</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>609870</th>\n",
              "      <td>1442786</td>\n",
              "      <td>684</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>851 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-11a0d473-0add-4e1d-a59a-f550661cf692')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
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
              "          document.querySelector('#df-11a0d473-0add-4e1d-a59a-f550661cf692 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-11a0d473-0add-4e1d-a59a-f550661cf692');\n",
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
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mostcount = tags_upd.groupby(['goodreads_book_id']).agg(\"sum\").sort_values(\"count\", ascending = 0).iloc[0].name\n",
        "mostcount"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fTxo4av_ozn9",
        "outputId": "2babeb2d-3a75-4da4-d306-2a9ef909b00a"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "865"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "myindex = books_upd[books_upd['goodreads_book_id'].isin([mostcount])].index\n",
        "myindex\n",
        "\n",
        "books_upd.iloc[myindex,:]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 226
        },
        "id": "Nn_RjlK8nBtr",
        "outputId": "6db1357c-3333-4161-e9cd-912210679e1b"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   book_id  goodreads_book_id  best_book_id  work_id  books_count      isbn  \\\n",
              "2       35                865           865  4835472          458  61122416   \n",
              "\n",
              "         isbn13                       authors  original_publication_year  \\\n",
              "2  9.780061e+12  Paulo Coelho, Alan R. Clarke                     1988.0   \n",
              "\n",
              "  original_title  ... work_ratings_count work_text_reviews_count  ratings_1  \\\n",
              "2   O Alquimista  ...            1403995                   55781      74846   \n",
              "\n",
              "   ratings_2  ratings_3  ratings_4  ratings_5  \\\n",
              "2     123614     289143     412180     504212   \n",
              "\n",
              "                                           image_url  \\\n",
              "2  https://images.gr-assets.com/books/1483412266m...   \n",
              "\n",
              "                                     small_image_url  NonEnglish  \n",
              "2  https://images.gr-assets.com/books/1483412266s...           0  \n",
              "\n",
              "[1 rows x 24 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-6e198246-b710-4b9d-a9a5-249701d04efd\">\n",
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
              "      <th>book_id</th>\n",
              "      <th>goodreads_book_id</th>\n",
              "      <th>best_book_id</th>\n",
              "      <th>work_id</th>\n",
              "      <th>books_count</th>\n",
              "      <th>isbn</th>\n",
              "      <th>isbn13</th>\n",
              "      <th>authors</th>\n",
              "      <th>original_publication_year</th>\n",
              "      <th>original_title</th>\n",
              "      <th>...</th>\n",
              "      <th>work_ratings_count</th>\n",
              "      <th>work_text_reviews_count</th>\n",
              "      <th>ratings_1</th>\n",
              "      <th>ratings_2</th>\n",
              "      <th>ratings_3</th>\n",
              "      <th>ratings_4</th>\n",
              "      <th>ratings_5</th>\n",
              "      <th>image_url</th>\n",
              "      <th>small_image_url</th>\n",
              "      <th>NonEnglish</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>35</td>\n",
              "      <td>865</td>\n",
              "      <td>865</td>\n",
              "      <td>4835472</td>\n",
              "      <td>458</td>\n",
              "      <td>61122416</td>\n",
              "      <td>9.780061e+12</td>\n",
              "      <td>Paulo Coelho, Alan R. Clarke</td>\n",
              "      <td>1988.0</td>\n",
              "      <td>O Alquimista</td>\n",
              "      <td>...</td>\n",
              "      <td>1403995</td>\n",
              "      <td>55781</td>\n",
              "      <td>74846</td>\n",
              "      <td>123614</td>\n",
              "      <td>289143</td>\n",
              "      <td>412180</td>\n",
              "      <td>504212</td>\n",
              "      <td>https://images.gr-assets.com/books/1483412266m...</td>\n",
              "      <td>https://images.gr-assets.com/books/1483412266s...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1 rows × 24 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-6e198246-b710-4b9d-a9a5-249701d04efd')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
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
              "          document.querySelector('#df-6e198246-b710-4b9d-a9a5-249701d04efd button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-6e198246-b710-4b9d-a9a5-249701d04efd');\n",
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
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "books_upd['original_title'][2]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "DtrDgMcdnGj5",
        "outputId": "d80e785d-b0d8-44ad-8100-300b17b99549"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'O Alquimista'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Q4: Which book (original_title) has the maximum number of ratings based on (work_ratings_count) [books.csv] ? (S)\n",
        "\n",
        "books_upd.sort_values(by='work_ratings_count', ascending = False).original_title[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "4nH-BE3DpL81",
        "outputId": "e5ac37b9-789d-4b14-eb7d-1b6ad9e5df75"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Angels & Demons'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Q5: Which (tag_id) is the most frequently used ie. mapped with the highest number of books [book_tags.csv] ? (In case of more than one tag, mention the tag id with the least numerical value) (N)\n",
        "\n",
        "d = dict()\n",
        "for i in tags_upd['tag_id']:\n",
        "  if i not in d.keys():\n",
        "    d[i] = 1\n",
        "  else:\n",
        "    d[i] = d[i] + 1\n",
        "\n",
        "#print(d)\n",
        "\n",
        "freq = -1\n",
        "my_id = 9999999999\n",
        "for key, value in d.items():\n",
        "  if value > freq:\n",
        "    freq = value\n",
        "    my_id = key\n",
        "  elif value == freq:\n",
        "    if key < my_id:\n",
        "      my_id = key\n",
        "\n",
        "print(my_id, freq)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SAJVG4W4vBY8",
        "outputId": "e99d25bc-6c16-46d0-c1f1-89797779d968"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "25647 851\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#### Q11: How many unique tags are there in the dataset [book_tags.csv] ? (N)\n",
        "\n",
        "len(tags_upd.tag_id.unique())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "89v-JNsy7eVD",
        "outputId": "3be0448b-990b-4614-fed7-f8a98663d11e"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "7905"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Q8: Which is the least used tag, i.e. mapped with the lowest number of books [book_tags.csv]? (In case of more than one tag, mention the tag id with the least numerical value) (N)\n",
        "\n",
        "# dictionary of tags is already created\n",
        "\n",
        "freq = 9999999999999\n",
        "my_id = 99999999999999\n",
        "\n",
        "for key, value in d.items():\n",
        "  if value < freq:\n",
        "    freq = value\n",
        "    my_id = key\n",
        "  elif value == freq:\n",
        "    if key < my_id:\n",
        "      my_id = key\n",
        "\n",
        "print(my_id, freq)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CPpEIg9D9wV6",
        "outputId": "1c8e22c6-4e14-4238-d4a5-252bd868900a"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "32 1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#### Q7: Which book (goodreads_book_id) is marked as to-read by most users [books.csv,toread.csv] ? (N)\n",
        "\n",
        "to_read_upd.to_excel('read.xlsx')"
      ],
      "metadata": {
        "id": "Vp2Iym8oqeeN"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "to_read_upd.groupby('book_id').book_id.agg('count').max() # row label 45"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N6drwXDtHdEf",
        "outputId": "301a9c92-c946-480a-c7c3-75940fbe81fd"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1717"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "books_upd[books_upd['book_id'] == 45]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 191
        },
        "id": "uIT6dzpHINDG",
        "outputId": "bf426587-c3ae-42c8-d31a-47a100b5b8f8"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   book_id  goodreads_book_id  best_book_id  work_id  books_count       isbn  \\\n",
              "5       45               4214          4214  1392700          264  770430074   \n",
              "\n",
              "         isbn13      authors  original_publication_year original_title  ...  \\\n",
              "5  9.780770e+12  Yann Martel                     2001.0     Life of Pi  ...   \n",
              "\n",
              "  work_ratings_count work_text_reviews_count  ratings_1  ratings_2  ratings_3  \\\n",
              "5            1077431                   42962      39768      74331     218702   \n",
              "\n",
              "   ratings_4  ratings_5                                          image_url  \\\n",
              "5     384164     360466  https://images.gr-assets.com/books/1320562005m...   \n",
              "\n",
              "                                     small_image_url  NonEnglish  \n",
              "5  https://images.gr-assets.com/books/1320562005s...           0  \n",
              "\n",
              "[1 rows x 24 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-92febc06-9190-4f63-a33c-170cf9fd3178\">\n",
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
              "      <th>book_id</th>\n",
              "      <th>goodreads_book_id</th>\n",
              "      <th>best_book_id</th>\n",
              "      <th>work_id</th>\n",
              "      <th>books_count</th>\n",
              "      <th>isbn</th>\n",
              "      <th>isbn13</th>\n",
              "      <th>authors</th>\n",
              "      <th>original_publication_year</th>\n",
              "      <th>original_title</th>\n",
              "      <th>...</th>\n",
              "      <th>work_ratings_count</th>\n",
              "      <th>work_text_reviews_count</th>\n",
              "      <th>ratings_1</th>\n",
              "      <th>ratings_2</th>\n",
              "      <th>ratings_3</th>\n",
              "      <th>ratings_4</th>\n",
              "      <th>ratings_5</th>\n",
              "      <th>image_url</th>\n",
              "      <th>small_image_url</th>\n",
              "      <th>NonEnglish</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>45</td>\n",
              "      <td>4214</td>\n",
              "      <td>4214</td>\n",
              "      <td>1392700</td>\n",
              "      <td>264</td>\n",
              "      <td>770430074</td>\n",
              "      <td>9.780770e+12</td>\n",
              "      <td>Yann Martel</td>\n",
              "      <td>2001.0</td>\n",
              "      <td>Life of Pi</td>\n",
              "      <td>...</td>\n",
              "      <td>1077431</td>\n",
              "      <td>42962</td>\n",
              "      <td>39768</td>\n",
              "      <td>74331</td>\n",
              "      <td>218702</td>\n",
              "      <td>384164</td>\n",
              "      <td>360466</td>\n",
              "      <td>https://images.gr-assets.com/books/1320562005m...</td>\n",
              "      <td>https://images.gr-assets.com/books/1320562005s...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1 rows × 24 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-92febc06-9190-4f63-a33c-170cf9fd3178')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
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
              "          document.querySelector('#df-92febc06-9190-4f63-a33c-170cf9fd3178 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-92febc06-9190-4f63-a33c-170cf9fd3178');\n",
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
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Q2: How many unique books are present in the dataset ? Evaluate based on the 'book_id'? [books.csv] (N)\n",
        "\n",
        "len(books_upd.book_id.unique())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KwVN1_xPMo-n",
        "outputId": "86c71e9c-7ca6-44d3-ea1c-75ae1249b2f6"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "833"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Q3: How many unique users are present in the dataset [ratings.csv] ? (N)\n",
        "len(ratings_upd.user_id.unique())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oyOP9GpCNhCx",
        "outputId": "36466b81-4fe5-466c-946f-8919a160c3af"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "50235"
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Q14: Predict sentiment using Textblob. How many positive titles (title) are there [books.csv] ? (cut-off >0) N\n",
        "\n",
        "# !pip install textblob\n",
        "\n",
        "from textblob import TextBlob\n",
        "\n",
        "def get_polarity(text):\n",
        "        return (TextBlob(text).polarity)\n",
        "\n",
        "books_upd[\"polarized\"] = books_upd[\"original_title\"].apply(get_polarity)\n",
        "books_upd[\"sentiments\"] = books_upd.polarized.apply(lambda x: \"positive\" if x > 0 else (\"negative\" if x < 0 else \"neutral\"))\n",
        "ans = books_upd[\"sentiments\"].value_counts()\n",
        "print(ans)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L9TS3DEUNrSh",
        "outputId": "09a1cf89-eabd-4f22-e274-a137f7ddec8e"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "neutral     639\n",
            "positive    120\n",
            "negative     74\n",
            "Name: sentiments, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## rough\n",
        "text = \" i think it`s under a honeymoon by the good life -- either that or it`s under a honey moon by joseph arthur.\"\n",
        "analysis = TextBlob(text)\n",
        "analysis.sentiment"
      ],
      "metadata": {
        "id": "HDYqHjHHNps0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4e0824e5-18e7-4d05-8015-838eff02d089"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Sentiment(polarity=0.7, subjectivity=0.6000000000000001)"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Z5vTbWIYHUKc"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}