from flask import Flask, render_template, redirect, request

import ml_model
app = Flask(__name__)


@app.route('/')
def html_file():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def classify():
    if request.method == 'POST':
        sepal_length = request.form['sepal_length']
        sepal_width = request.form['sepal_width']

        petal_length = request.form['petal_length']
        petal_width = request.form['petal_width']

        class_name = ml_model.predict(
            sepal_length, sepal_width, petal_length, petal_width)
        print(class_name)
    return render_template("index.html", your_class=class_name)


if __name__ == '__main__':
    app.run(debug=True)
