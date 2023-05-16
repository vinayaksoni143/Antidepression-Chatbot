import customtkinter as ctk
from chat import chat_eval

root=ctk.CTk()
root.geometry("600x550")
root.title("Chatbot")
title_label=ctk.CTkLabel(root,text="Anti Depression Chatbot",font=ctk.CTkFont(size=22,weight="bold"))

title_label.pack(padx=10,pady=(40,20))

def send_message():
    user_input = entry_field.get()

    label=ctk.CTkLabel(chat_display,text="You : "+ user_input + "\n",font=ctk.CTkFont(size=16),justify="left")
    label.pack(anchor="w")
    
    

    if user_input.lower() == "exit":
        label=ctk.CTkLabel(chat_display,text="Chatbot : Goodbye!",font=ctk.CTkFont(size=16))
        label.pack(anchor="w",padx=5)
        root.destroy()  # Close the chatbot GUI window
        return
    
    chatbot_response=chat_eval(user_input)
    label=ctk.CTkLabel(chat_display,text="Chatbot : " + chatbot_response + "\n",font=ctk.CTkFont(size=16),wraplength=550,justify="left")
    label.pack(anchor="w")

    

    

    entry_field.delete(0,ctk.END)

def combined_function():
    send_message()
    scroll_to_bottom()


def scroll_to_bottom():
    chat_display.yview_moveto(1.0)

chat_display = ctk.CTkScrollableFrame(root, width=550, height=400)
chat_display.pack()


input_fields=ctk.CTkFrame(root,width=550,height=60)
input_fields.pack(pady=10)

entry_field = ctk.CTkEntry(input_fields, width=500,height=30)
entry_field.grid(row=0,column=0)

send_button=ctk.CTkButton(input_fields,text="Send",width=50,command=combined_function)
send_button.grid(row=0,column=1,padx=10)


root.mainloop()