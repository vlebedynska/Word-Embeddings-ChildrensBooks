from bs4 import BeautifulSoup
import re


def createTabSeparatedBooksList():
    books_extracted_list = []
    books_extracted = open("corpus/books_extracted.txt", "w")
    bibliog = open("corpus/bibliog.html");
    soup = BeautifulSoup(bibliog, "html.parser")
    with open("corpus/books.txt", "r") as text_file:
        for line in text_file:
            abbreviation_list = line.split(',')
            books_extracted_list.append(abbreviation_list[0])
            try:
                book_name = soup.find("span", text=abbreviation_list[0])
                book_name_parent = book_name.parent.text
                book_name_parent = book_name_parent.replace("\n", " ")
                id, wordcnt, title, year = re.match("^\s*([\dA-Z]{3}) (\d+) words from (.+) (\d+\s)", book_name_parent).groups()
                books_extracted.write(id +";" + wordcnt + ";" + title + ";" + year + ";\n")
                print(book_name_parent)
            except:
                "Not found."
    books_extracted.close()
    bibliog.close()



if __name__ == '__main__':
    createTabSeparatedBooksList()
