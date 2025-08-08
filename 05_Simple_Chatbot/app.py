import chainlit as cl

@cl.on_message

async def main(message: cl.Message): # Message is the class for chainlit this class use for conversation

    responce = f'You Said: {message.content}' #  content : user jo bhi khy ga wo autotically content me chala jata hai

    await cl.Message(content=responce).send() # responce k under over all conversation store hoti hai :
                                 # # send() method is used to send the message to the user


    