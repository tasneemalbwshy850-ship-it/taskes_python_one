from member import Member
class staff_member(Member):
    def __init__(self, staff_id, book,):
        self.staff_id = staff_id
        self.book = book
        

    def add_book(self, title, author, ISBN, available):
        new_book = {
            "title": title,
            "author": author,
            "ISBN": ISBN,
            "available": available
        }
        self.book.append(new_book)
        print(f"Book '{title}' added to the library.")
        return self.book
    
