# A Mean-Reversion Based Algorithmic Trading Strategy

The contents of this folder contain a number of developed pieces of code as part of the final year project for the MEng Computer Science course at UCL. The contents are split as follows:

- historic_data: Contains two different CSV files containing the historic price data for a number of foreign exchange pairs. These are mainly supplied for the use by notebooks in the Jupyter Notebooks subfolder.
	- "10 Year Data" - Contains past price information for 13 different foreign exchange pairs, from January 1st 2010 to January 1st 2020.
	- "20 Year Data" - Contains past price information for GBP/USD and EUR/GBP pairs, from January 1st 2000 to 1st January 2020.

- Jupyter Notebooks: Contains a number of .ipynb files (notebooks) populated with developed code used to perform the analyses we reference within the paper. Source code for our developed strategy and the other simple strategies is available in these notebooks. This code is thoroughly documented and should accurately describe the intended behaviours of each strategy.

- oup_evaluate.py: Standalone python script intended to demonstrate how the developed strategy may be used as a larger part of a fully automated trading strategy. As it is, this script allows for the full customisation of the parameters of our strategy from a command line interface. In order to extract real-time information, we make use of the Alpha-Vantage API, which comes with limits, but since we add this for demonstration purposes, it should suffice. 
	- Requirements: pandas, numpy, scipy, alpha-vantage
	- Usage: use "python oup\_evaluate.py -h" at the command line for details on command line usage. The only required arguments are FROM and TO, which correspond to trading the currency pair FROM/TO. For example, using "python oup\_evaluate.py USD GBP" at the command line will provide a real-time recommendation for buying or selling USD/GBP pairs, using the developed OUP strategy. 
	- Support: the standalone script relies on the Alpha-Vantage API, and as such is capable of evaluating any pairs the API supports. See https://www.alphavantage.co/documentation/ for details.
	- Expected Output: the script produces a line of console feedback, stating if it would currently sell or buy the pair. In the event that it suggests a buy, it states how long ago it would have executed the buy order. 