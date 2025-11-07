# write a python script to download a file from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&is_datatable=true&acc=GPL13534&id=11288&db=GeoDb_blob92
# and save it to the ./dataset directory with the filename GPL13534-11288.txt
import os
import requests


def download_file(url, output_path):
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    with open(output_path, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded and saved to {output_path}")


if __name__ == "__main__":
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&is_datatable=true&acc=GPL13534&id=11288&db=GeoDb_blob92"
    output_dir = "./dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "GPL13534-11288.txt")
    download_file(url, output_path)
