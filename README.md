# License Plate Recognition System for Motorcycles

This project is a specialized License Plate Recognition (LPR) system tailored for detecting license plates on motorcycles using computer vision and deep learning techniques. The system is capable of recognizing characters on license plates in motorcycle images, extracting the plate number, and saving the results to a database.

## How to Use

1. Clone the repository:

```
git clone https://github.com/roniragilimankhoirul/License-Plate-Recognition-System-for-Motorcycles.git
 && cd License-Plate-Recognition-System-for-Motorcycles
```

2. Create a virtual environment:

```
python -m venv myenv
```

3. Activate the virtual environments:

```
source myenv/bin/activate
```

4. Install Dependencies:

```
pip install -r requirements.txt
```

5. Set up your environment by creating a .env file with the following variables:

```
DB_HOST=your_database_host
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_NAME=your_database_name
```

6. Training model:

```
python training.py
```

7. Run the image detection program and save to database:

```
python main.py <Image_Path>
```

### Example outputs

![lpd](./output_example/AA5627JT_output.jpg)
