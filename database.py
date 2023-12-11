import mysql.connector
import os
from dotenv import load_dotenv
import base64
import io

load_dotenv()

def connect_to_database():
    conn = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )
    return conn

def create_plate_number_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plate_number (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number TEXT,
            image_path VARCHAR(255)
        )
    ''')

def save_to_database(plate_number, img_base64):
    try:
        conn = connect_to_database()
        cursor = conn.cursor()

        create_plate_number_table(cursor)

        # Generate a sanitized filename using the plate number
        sanitized_plate_number = "".join(c if c.isalnum() else "_" for c in plate_number)
        filename = f"result/{sanitized_plate_number}_output.jpg"
        filepath = os.path.join(os.getcwd(), filename)

        # Create the result directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the image to the file
        img = base64.b64decode(img_base64)
        with open(filepath, 'wb') as file:
            file.write(img)

        # Save the result to the database
        cursor.execute('INSERT INTO plate_number (plate_number, image_path) VALUES (%s, %s)', (plate_number, filename))

        conn.commit()
        print("Data saved to the database and file written successfully")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Your existing code for license plate recognition

    # Assuming you have extracted the plate number into the variable 'plate_number'
    # and the final image is in the variable 'img_show_plate'

    # Convert the image to base64
    img_pil = Image.fromarray(img_show_plate)
    img_byte_array = io.BytesIO()
    img_pil.save(img_byte_array, format='JPEG')
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

    # Save the result to the database
    save_to_database(plate_number, img_base64)
