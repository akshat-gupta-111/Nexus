from server import mcp
from utils.file_reader import read_csv_summary

@mcp.tool()
def summarize_csv_file(filename : str) -> str:
    """
    Summarize CSV file by returning its row and column count.
    Args:
        filename : Name of the CSV file (e.g. 'sample.csv')
    Returns:
        A String containing row and column count.
    """
    return read_csv_summary(filename)