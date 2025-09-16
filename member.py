class Member:
    def __init__(self,name,membership_id,list_of_borrowed_books,return_books):
        self.name = name
        self.set__membership_id = membership_id
        self.list_of_borrowed_books = list_of_borrowed_books
        self.return_books = return_books
    def borrow_book(self):
        if self.list_of_borrowed_books == True :
            print("the book is available for borrowing")
        else : 
            print("the book is not available")

    def return_book(self):
        if self.return_books == True:
            print("the book has been returned")
        else:
            print("the book was not borrowed")



