import re
import math


def print_dictionary(dictionary):
    for ell in list(dictionary.items()):
        print(f"{ell[0]} - {ell[1]}")


def sep_line(val=None, ch="_", length=50):
    if val:
        str_len = len(val)
        one_line_length = (length - str_len - 2) // 2
        print(f"{one_line_length*ch} {val} {one_line_length*ch}")
    else:
        print(length * ch)


################ one-hot encoding ################


def create_dictionary(doc1, doc2):
    words_doc1 = re.findall(r"\w+", doc1.lower())
    words_doc2 = re.findall(r"\w+", doc2.lower())

    words = sorted(set(words_doc1) | set(words_doc2))

    dictionary = {}

    words_amount = len(words)
    for index in range(words_amount):
        vector = [0] * words_amount
        vector[index] = 1
        dictionary[words[index]] = vector

    return dictionary


def one_hot_encoding(doc, dictionary):
    doc_encoded = {}
    d = re.findall(r"\w+", doc.lower())
    dictionary_list = list(dictionary.items())
    dictionary_size = len(dictionary_list)
    for ell in dictionary_list:
        if ell[0] in d:
            doc_encoded[ell[0]] = ell[1]
        else:
            doc_encoded[ell[0]] = [0] * dictionary_size

    return doc_encoded


##################################################
################## bag of words ##################


def bag_of_words(doc1, doc2):
    words_doc1 = re.findall(r"\w+", doc1.lower())
    words_doc2 = re.findall(r"\w+", doc2.lower())

    words = sorted(set(words_doc1) | set(words_doc2))

    words_size = len(words)

    doc1_encoded = [0] * words_size
    doc2_encoded = [0] * words_size

    for i in range(words_size):
        if words[i] in words_doc1:
            doc1_encoded[i] = 1
        if words[i] in words_doc2:
            doc2_encoded[i] = 1

    return doc1_encoded, doc2_encoded


##################################################
##################### tf_idf #####################


def unique_sorted_words(list_of_lists):
    unique_words = set(word for sublist in list_of_lists for word in sublist)
    sorted_unique_words = sorted(unique_words)
    return sorted_unique_words


def documents_split(documents: list):
    result = []
    for document in documents:
        result.append(re.findall(r"\w+", document.lower()))

    return result


def count_documents_with_word(documents, word):
    count = 0
    for doc in documents:
        if word in doc:
            count += 1
    return count


def tf(words: list, word: str):
    nt = words.count(word)
    return nt / len(words)


def idf(documents, word):
    word_count = count_documents_with_word(documents=documents, word=word)
    documents_amount = len(documents)
    return math.log((documents_amount / word_count))


def tf_idf(documents: list):
    documents_splited = documents_split(documents)
    all_words = unique_sorted_words(documents_splited)

    for document in documents_splited:
        print(f"\n{document}")
        for word in all_words:
            tf_val = tf(document, word)
            idf_val = idf(documents_splited, word)
            print(f"{word}: TF: {tf_val}, IDF: {idf_val}, TF-IDF: {tf_val*idf_val}")

    return


##################################################


# doc1 = "Пес сел на пень"
# doc2 = "Кот сел на ель"

doc1 = "Кот гуляет по улице."
doc2 = "По крыше гуляет ворон."


dictionary = create_dictionary(doc1=doc1, doc2=doc2)

sep_line("one-hot encoding")
print("Dictionary")
print_dictionary(dictionary)

print("\ndoc1 encoded")
print(doc1)
print_dictionary(one_hot_encoding(doc1, dictionary))

print("\ndoc2 encoded")
print(doc2)
print_dictionary(one_hot_encoding(doc2, dictionary))

sep_line()

sep_line("bag of words")


bow_d1, bow_d2 = bag_of_words(doc1=doc1, doc2=doc2)

print(f"{doc1} - {bow_d1}")
print(f"{doc2} - {bow_d2}")

sep_line()

sep_line("tf_idf")

tf_idf([doc1, doc2])
sep_line()


"""
________________ one-hot encoding ________________
Dictionary
ворон - [1, 0, 0, 0, 0, 0]
гуляет - [0, 1, 0, 0, 0, 0]
кот - [0, 0, 1, 0, 0, 0]
крыше - [0, 0, 0, 1, 0, 0]
по - [0, 0, 0, 0, 1, 0]
улице - [0, 0, 0, 0, 0, 1]

doc1 encoded
Кот гуляет по улице.
ворон - [0, 0, 0, 0, 0, 0]
гуляет - [0, 1, 0, 0, 0, 0]
кот - [0, 0, 1, 0, 0, 0]
крыше - [0, 0, 0, 0, 0, 0]
по - [0, 0, 0, 0, 1, 0]
улице - [0, 0, 0, 0, 0, 1]

doc2 encoded
По крыше гуляет ворон.
ворон - [1, 0, 0, 0, 0, 0]
гуляет - [0, 1, 0, 0, 0, 0]
кот - [0, 0, 0, 0, 0, 0]
крыше - [0, 0, 0, 1, 0, 0]
по - [0, 0, 0, 0, 1, 0]
улице - [0, 0, 0, 0, 0, 0]
__________________________________________________
__________________ bag of words __________________
Кот гуляет по улице. - [0, 1, 1, 0, 1, 1]
По крыше гуляет ворон. - [1, 1, 0, 1, 1, 0]
__________________________________________________
_____________________ tf_idf _____________________

['кот', 'гуляет', 'по', 'улице']
ворон: TF: 0.0, IDF: 0.6931471805599453, TF-IDF: 0.0
гуляет: TF: 0.25, IDF: 0.0, TF-IDF: 0.0
кот: TF: 0.25, IDF: 0.6931471805599453, TF-IDF: 0.17328679513998632
крыше: TF: 0.0, IDF: 0.6931471805599453, TF-IDF: 0.0
по: TF: 0.25, IDF: 0.0, TF-IDF: 0.0
улице: TF: 0.25, IDF: 0.6931471805599453, TF-IDF: 0.17328679513998632

['по', 'крыше', 'гуляет', 'ворон']
ворон: TF: 0.25, IDF: 0.6931471805599453, TF-IDF: 0.17328679513998632
гуляет: TF: 0.25, IDF: 0.0, TF-IDF: 0.0
кот: TF: 0.0, IDF: 0.6931471805599453, TF-IDF: 0.0
крыше: TF: 0.25, IDF: 0.6931471805599453, TF-IDF: 0.17328679513998632
по: TF: 0.25, IDF: 0.0, TF-IDF: 0.0
улице: TF: 0.0, IDF: 0.6931471805599453, TF-IDF: 0.0
__________________________________________________
"""
