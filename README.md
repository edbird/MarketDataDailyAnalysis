# MarketDataDailyAnalysis
Equities Markets Daily Data Statistics Analysis

# What is this project all about?

I wanted to do some experimental data analysis in Python inspired in part by the analysis work which I did during my PhD.

This was an opportunity for me to learn how to:

- perform this analysis using Python libraries (my PhD work was done using C++ and a library `CERN ROOT`)
- revise some Statistical Analysis Methods
- add a Python project to my CV


# Data Origins

I intended to perform some analysis using closing market price data from the stock market. Since price (and other) data on the subject of financial instruments is quite hard to obtain without expensive licenses, I opted for some free data which I was able to obtain from eodhd.com.

The data contains daily closing prices for the AAPL stock ticker. Other equities instruments are available, I opted for Apple stock by default.

I wanted to assess how the changes in daily closing prices are distributed.

- While the distribution of daily changes in closing price is **not Gaussian distributed**, for AAPL stock for the last year (250 trading days), **the distribution is quite well modelled by a Gaussian**.
- The purpose of this project is **not** to model the *distribution*, but to *demonstrate the application of some statistical methods in a financial markets context*.


# Statistical Methods

I use a few methods from Statistics, ranging from very simple measurements of statistics such as the mean and variance of some data, to more complex methods which use minimization algorithms to perform Chi-Squared (least-squares) fitting, as well as Maximum Likelihood methods, both binned and unbinned.

The results from Least-Squares and Binned Maximum Likelihood (ML) model fitting are biased. The unbinned Maximum Likelihood estimation method is used for the majority of the analysis, as it is an unbiased estimator. The other methods, Chi-Square and binned ML are only used to compared results.


# MongoDB

The experimental data results are stored in a MongoDB instance, with each experiment represented by a BSON (binary JSON) document.

# Big Data and Data Generation at Scale

With MongoDB as a backend for data storage and management, it is easy to scale into the realms of big data.

The analysis was performed on a 16-thread AMD Ryzen 7 5700G processor. 10 threads were used throughout to leave some processing power available for other projects. With this configuration, it was possible to perform 4000 unbinned Maximum Liklihood measurements per 60 seconds.

**insert number of records here**
