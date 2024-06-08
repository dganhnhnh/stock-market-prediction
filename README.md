# Stock market prediction
A group projects on implementing and evaluating machine learning algorithms to predict stock price bases on stock historical data only. 

## About the project 
This work is a group projects offered in course *IT3190-Introduction to Machine Learning and Data Mining* of ***Hanoi University of Science and Technology***.
We used 
- Tree-based methods (Random forest, Gradient boosting), 
- Artificial neural net based-method (Mulit layer perceptron with Back propagation, Long-short term memory),
- Support vector regressor with $\tanh$ kernel with parameters optimized by Pacticle swarm optimization.

### Built with
- Python
- Scikit-learn
- Tensorflow
- Pytorch
## Getting Started
### Prerequisites
- [Install Python ](https://www.python.org/downloads/)
### Installation
1. Clone the repository
2. Install python libraries in Virtual environment
```
$ pip install yfinance
$ pip install scikit-learn
$ pip install matplotlib
$ pip install seaborn
$ pip install torch
$ pip install keras
$ pip install tensorflow
$ pip install pandas
```
3. Get data: Run CollectData.ipynb or get data directly from ![yahoo!Finance](https://finance.yahoo.com/)

## Usage
- Run .ipynb files directly or use python scripts.
- Using python scripts:
1. Create a new .py file, eg: test.py.
2. Import model you want to use from python scripts, eg:
```
# In test.py

from ANN_BP import ANN_BP_Model as Model
```
3. Use methods from Model to train, eg:
```
# In test.py
ticker = 'GOOGL'

model = Model()
model.prepare_data(ticker)
model.train()
model.calculate_loss()
model.plot_result()
``` 


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact
Project Link: https://github.com/dganhnhnh/stock-market-prediction

## References
Materials:

Hegazy, Osman, Omar S. Soliman, and Mustafa Abdul Salam. "A machine learning model for stock market prediction." arXiv preprint arXiv:1402.7351 (2013). Web. https://arxiv.org/pdf/1402.7351.

Li, Bo, and Xitian Tian. "An effective PSO-LSSVM-based approach for surface roughness prediction in high-speed precision milling." IEEE Transactions on Instrumentation and Measurement (2021). Web. https://ieeexplore.ieee.org/document/9443228.

Scikit-learn developers. "SVR: 1.4.7.2. SVR." scikit-learn 1.0.2 documentation,
https://scikit-learn.org/stable/modules/svm.html#svr .

Bài giảng Nhập môn Học máy và Khai phá dữ liệu (IT3190)
Vu, S. "Ensemble Learning trong Machine Learning: Boosting, Bagging,
Stacking (Sử dụng R code)." https://svcuong.github.io/post/ensemble-learning/

Bài giảng Nhập môn Học máy và Khai phá dữ liệu (IT3190)
