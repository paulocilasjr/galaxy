import argparse
import json
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.table as table

# Helper function to convert string values to appropriate types
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

# Model class to store model information
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

# Abstract base class for file parsers
class FileParser(ABC):
    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def parse(self):
        pass

# Parser for HTML files
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

# Parser for JSON files
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

# Class to extract metrics from input files
class MetricsExtractor:
    def __init__(self, file_inputs):
        self.file_inputs = file_inputs  # list of (file_path, file_type)

    def extract(self):
        results = []
        for file_path, file_type in self.file_inputs:
            parser = self.get_parser(file_path, file_type)
            if parser:
                try:
                    models = parser.parse()
                    results.append(models)
                except Exception as e:
                    print(f"Error parsing {file_path}: {str(e)}")
            else:
                print(f"Unsupported file type: {file_type} for {file_path}")
        return results

    def get_parser(self, file_path, file_type):
        if file_type == 'html':
            return HTMLParser(file_path)
        elif file_type == 'json':
            return JSONParser(file_path)
        else:
            return None

# Main execution block
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate SVG figure from model metrics")
    parser.add_argument('--input_json', type=str, help="Path to input JSON file")
    parser.add_argument('--input_html', type=str, help="Path to input HTML file")
    parser.add_argument('--output_svg', type=str, required=True, help="Path to output SVG file")
    args = parser.parse_args()

    # Collect input files
    file_inputs = []
    if args.input_json:
        file_inputs.append((args.input_json, 'json'))
    if args.input_html:
        file_inputs.append((args.input_html, 'html'))

    # Check if any input files are provided
    if not file_inputs:
        print("No input files provided.")
        exit(1)

    # Extract models from input files
    extractor = MetricsExtractor(file_inputs)
    all_models_dicts = extractor.extract()
    all_models = {}
    for models_dict in all_models_dicts:
        all_models.update(models_dict)

    # Check if any models were found
    if not all_models:
        print("No models found in the input files.")
        exit(1)

    # Collect all unique metric names
    all_metric_names = set()
    for model in all_models.values():
        all_metric_names.update(model.metrics.keys())
    all_metric_names = sorted(list(all_metric_names))

    # Create table data
    table_data = []
    for model_name, model in all_models.items():
        row = [model_name]
        for metric in all_metric_names:
            value = model.metrics.get(metric, 'N/A')
            row.append(str(value))  # Convert to string for table
        table_data.append(row)

    # Add header
    header = ['Model Name'] + all_metric_names
    table_data.insert(0, header)

    # Create figure and table
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(all_models))))
    ax.axis('tight')
    ax.axis('off')
    the_table = table.table(ax, cellText=table_data, loc='center', cellLoc='center', edges='closed')

    # Save as SVG
    fig.savefig(args.output_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
