from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('bet_game.html')

@app.route('/results', methods=['POST'])
def calculate():
    if request.method == 'POST':
        number = float(request.form['number'])
        result = 2 * number
        return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
