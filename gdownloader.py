import gdown
import sys

if __name__ == "__main__":
    if len(sys.argv) is not 3:
        print("Usage: python gdownloader.py drive_file_id destination_file_path")
    url = sys.argv[1]#'https://drive.google.com/uc?id=1AcN5g_fBn94Xq9Cp9q3WKDkAG9WTiXUF'
    output = sys.argv[2]#'test.zip'
    gdown.download(url, output, quiet=False)