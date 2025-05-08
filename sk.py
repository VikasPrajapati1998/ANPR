from flask import Flask, request

app = Flask(__name__)

@app.route('/vehicle_entry', methods=['POST'])
def vehicle_entry():
    plate_number = request.form.get('plate_number')
    vehicle_type = request.form.get('vehicle_type')
    plate_image = request.files.get('plate_image')
    if plate_number and vehicle_type and plate_image:
        # Process the data here
        print(plate_number )
        print(vehicle_type)
        print(plate_image)
        return 'Data received', 200
    else:
        return 'Missing data', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
