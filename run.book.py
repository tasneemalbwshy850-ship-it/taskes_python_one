from Book import Book
from member import Member 
from staff_member import staff_member

book1 = Book("1984", "George Orwell", "1234567890", True)
book2 = Book("Dune","Frank Herbert","978_0441172719", False)
book3 = Book("The Hobbit","J.R.R. Tolkien","978_0547928227", True)
book4 = Book("Fahrenheit 451","Ray Bradbury","978_1451673319", True)
book5 = Book("Brave New World","Aldous Huxley","978_0060850524", False)

member1 =  Member("Alice","M001",True,False)
member2 =  Member("Bob","M002",False,True)
member3 =  Member("Charlie","M003",True,True)
member4 =  Member("David","M004",False,False)
member5 =  Member("Eve","M005",True,False)


print(book1.title,book2.author,book3.get__ISBN,book4.title,book5.available)  
print(member1.return_books,member2.list_of_borrowed_books,member5.name,member3.set__membership_id)

add_book = staff_member("S001", [])

add_book.add_book("The Catcher in the Rye","J.D. Salinger","978_0316769488",True)
add_book.add_book("To Kill a Mockingbird","Harper Lee","978_0061120084",True)





