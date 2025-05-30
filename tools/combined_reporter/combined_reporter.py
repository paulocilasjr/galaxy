"""
developing
"""
import argparse
import os
import json
import csv
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

def convert_value(value_str):
    if value_str.lower() == 'none':
        return None
    elif value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str

class Model:
    def __init__(self, name, metrics=None, hyperparameters=None, test_set_results=None):
        self.name = name
        self.metrics = metrics or {}
        self.hyperparameters = hyperparameters or {}
        self.test_set_results = test_set_results or {}

    def to_dict(self):
        return {
            "name": self.name,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "test_set_results": self.test_set_results
        }

class FileParser(ABC):
    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def parse(self):
        pass

class HTMLParser(FileParser):
    def parse(self):
        with open(self.file_path, 'r') as f:
            soup = BeautifulSoup(f, 'html.parser')
        hyperparameters = self.extract_hyperparameters(soup)
        comparison_results = self.extract_comparison_results(soup)
        test_set_results = self.extract_test_set_results(soup)
        best_model_name = list(test_set_results.keys())[0] if test_set_results else None
        models = {}
        for model_name, metrics in comparison_results.items():
            model = Model(name=model_name, metrics=metrics)
            if model_name == best_model_name:
                model.hyperparameters = hyperparameters
                model.test_set_results = test_set_results[model_name]
            models[model_name] = model
        return models

    def extract_hyperparameters(self, soup):
        table = soup.find('table', class_='dataframe table')
        if table:
            hyperparameters = {}
            for row in table.find('tbody').find_all('tr'):
                cells = row.find_all('td')
                if len(cells) == 2:
                    param = cells[0].text.strip()
                    value_str = cells[1].text.strip()
                    hyperparameters[param] = convert_value(value_str)
            return hyperparameters
        return {}

    def extract_comparison_results(self, soup):
        h2 = soup.find('h2', string="Comparison Results on the Cross-Validation Set")
        if h2:
            table = h2.find_next('table', class_='dataframe table')
            if table:
                headers = [th.text.strip() for th in table.find('thead').find('tr').find_all('th')]
                comparison_results = {}
                for row in table.find('tbody').find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) == len(headers):
                        model_name = cells[0].text.strip()
                        metrics = {headers[i]: convert_value(cells[i].text.strip()) for i in range(1, len(headers))}
                        comparison_results[model_name] = metrics
                return comparison_results
        return {}

    def extract_test_set_results(self, soup):
        h2 = soup.find('h2', string="Results on the Test Set for the best model")
        if h2:
            table = h2.find_next('table', class_='dataframe table')
            if table:
                headers = [th.text.strip() for th in table.find('thead').find('tr').find_all('th')]
                test_set_results = {}
                for row in table.find('tbody').find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) == len(headers):
                        model_name = cells[0].text.strip()
                        metrics = {headers[i]: convert_value(cells[i].text.strip()) for i in range(1, len(headers))}
                        test_set_results[model_name] = metrics
                return test_set_results
        return {}

class JSONParser(FileParser):
    def parse(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        models = {}
        for model_name, model_data in data.items():
            model = Model(
                name=model_name,
                metrics=model_data.get("metrics", {}),
                hyperparameters=model_data.get("hyperparameters", {}),
                test_set_results=model_data.get("test_set_results", {})
            )
            models[model_name] = model
        return models

class CSVParser(FileParser):
    def parse(self):
        models = {}
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row["Model"]
                metrics = {k: convert_value(v) for k, v in row.items() if k != "Model"}
                model = Model(name=model_name, metrics=metrics)
                models[model_name] = model
        return models

class MetricsExtractor:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def extract(self):
        results = []
        for file_path in self.file_paths:
            parser = self.get_parser(file_path)
            if parser:
                models = parser.parse()
                results.append(models)
            else:
                print(f"Unsupported file type for {file_path}")
        return results

    def get_parser(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.html':
            return HTMLParser(file_path)
        elif ext == '.json':
            return JSONParser(file_path)
        elif ext == '.csv':
            return CSVParser(file_path)
        else:
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model metrics from files")
    parser.add_argument('file_paths', nargs='+', help="Paths to the input files (HTML, JSON, CSV)")
    args = parser.parse_args()

    extractor = MetricsExtractor(args.file_paths)
    results = extractor.extract()

    for i, models in enumerate(results):
        print(f"File {i+1}:")
        for model_name, model in models.items():
            print(f"  Model: {model_name}")
            print(f"    Metrics: {model.metrics}")
            if model.hyperparameters:
                print(f"    Hyperparameters: {model.hyperparameters}")
            if model.test_set_results:
                print(f"    Test Set Results: {model.test_set_results}")
