#include "CSVDataHandler.h"

// Constructor
CSVDataHandler::CSVDataHandler(int dimension) : _dimension(dimension)
{
    if (_dimension != 2 && _dimension != 3)
    {
        throw std::invalid_argument("Dimension must be 2 or 3");
    }

    // Initialize headers based on the dimension
    _time_header = "time";

    _E_headers = (_dimension == 2) ? std::vector<std::string>{"E11", "E22", "E12"}
                                   : std::vector<std::string>{"E11", "E22", "E33", "E23", "E13", "E12"};

    _S_headers = (_dimension == 2) ? std::vector<std::string>{"S11", "S22", "S12"}
                                   : std::vector<std::string>{"S11", "S22", "S33", "S23", "S13", "S12"};

    // _C_headers = (_dimension == 2) ? std::vector<std::string>{"C1111", "C1122", "C1112", "C2222", "C2212", "C1212"}
    //                                : std::vector<std::string>{"C1111", "C1122", "C1133", "C1123", "C1113", "C1112",
    //                                                           "C2222", "C2233", "C2223", "C2213", "C2212", "C3333",
    //                                                           "C3323", "C3313", "C3312", "C2323", "C2313", "C2312",
    //                                                           "C1313", "C1312", "C1212"};

    _all_headers = std::vector<std::string>{_time_header};
    _all_headers.insert(_all_headers.end(), _E_headers.begin(), _E_headers.end());
    _all_headers.insert(_all_headers.end(), _S_headers.begin(), _S_headers.end());
    // _all_headers.insert(_all_headers.end(), _C_headers.begin(), _C_headers.end());

    _abs_max_E = 0;
    _abs_max_S = 0;
    // _abs_max_C = 0;
}

/**
 * @param directoryPath directory path of data
 * @param dimension dimension of the problem 2 or 3
 * @param enable_padding Whether to pad the initial state to make all sequences the same length
 */
CSVDataHandler::CSVDataHandler(const std::string &directoryPath, int dimension, unsigned int min_length, const bool enable_padding)
    : CSVDataHandler(dimension)
{
    readAndStoreData(directoryPath);

    filterShortSeries(min_length);

    if (enable_padding)
        padInitialStates();
}

// Read and store all CSV files in the directory
void CSVDataHandler::readAndStoreData(const std::string &directoryPath)
{
    for (const auto &entry : std::filesystem::directory_iterator(directoryPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".csv")
        {
            readCSVFile(entry.path().string(), false);
        }
    }
    findMax();
}

// Read a single CSV file
void CSVDataHandler::readCSVFile(const std::string &filePath, bool find_max)
{
    rapidcsv::Document doc(filePath);

    // Check if all necessary headers are present
    if (!checkHeaders(doc))
    {
        std::cerr << "Skipping file due to missing headers: " << filePath << std::endl;
        return;
    }

    // Clear current file data before reading new data
    _current_file_time.clear();
    _current_file_E.clear();
    _current_file_S.clear();
    // _current_file_C.clear();

    // Read the time column
    _current_file_time = doc.GetColumn<double>(_time_header);
    if (_current_file_time.size() > 1)
        _current_file_time.erase(_current_file_time.begin());

    // Read E, S, and C columns
    _current_file_E = readMultipleColumns(doc, _E_headers);
    _current_file_S = readMultipleColumns(doc, _S_headers);
    // _current_file_C = readMultipleColumns(doc, _C_headers);

    // Save the current file's data to the overall storage
    _times.push_back(_current_file_time);
    _Es.push_back(_current_file_E);
    _Ss.push_back(_current_file_S);
    // _Cs.push_back(_current_file_C);

    if (find_max)
        findMax();
}

// Helper function to check for necessary headers
bool CSVDataHandler::checkHeaders(rapidcsv::Document &doc)
{
    // Retrieve all column names from the document
    const std::vector<std::string> &column_names = doc.GetColumnNames();

    for (const auto &header : _all_headers)
    {
        if (std::find(column_names.begin(), column_names.end(), header) == column_names.end())
        {
            std::cerr << "Missing header: " << header << std::endl;
            return false;
        }
    }

    return true;
}

// Helper function to read multiple columns based on headers
std::vector<std::vector<double>> CSVDataHandler::readMultipleColumns(rapidcsv::Document &doc, const std::vector<std::string> &headers)
{
    std::vector<std::vector<double>> columns;

    for (const auto &header : headers)
    {
        std::vector<double> column_data = doc.GetColumn<double>(header);

        // Check if the column has more than one row (header and data row)
        if (column_data.size() > 1)
        {
            // Skip the first data row by erasing the first element after the header
            column_data.erase(column_data.begin());
        }

        columns.push_back(column_data);
    }

    return columns;
}

void CSVDataHandler::filterShortSeries(unsigned int min_length)
{
    bool changed = false;
    for (unsigned int i = 0; i < _times.size(); i++)
    {
        if (_times[i].size() < min_length)
        {
            _times.erase(_times.begin() + i);
            _Es.erase(_Es.begin() + i);
            _Ss.erase(_Ss.begin() + i);
            // _Cs.erase(_Cs.begin() + i);
            changed = true;
        }
    }

    if (changed)
        findMax();
}

void CSVDataHandler::findMax()
{
    _abs_max_E = 0;
    _abs_max_S = 0;
    // _abs_max_C = 0;
    for (const auto &file_E : _Es)
        for (const auto &col_E : file_E)
        {
            for (const auto &el_E : col_E)
            {
                double absValue = std::abs(el_E);
                if (absValue > _abs_max_E)
                    _abs_max_E = absValue;
            }
        }

    for (const auto &file_S : _Ss)
        for (const auto &col_S : file_S)
        {
            for (const auto &el_S : col_S)
            {
                double absValue = std::abs(el_S);
                if (absValue > _abs_max_S)
                    _abs_max_S = absValue;
            }
        }

    // for (const auto &file_C : _Cs)
    //     for (const auto &col_C : file_C)
    //     {
    //         for (const auto &el_C : col_C)
    //         {
    //             double absValue = std::abs(el_C);
    //             if (absValue > _abs_max_C)
    //                 _abs_max_C = absValue;
    //         }
    //     }
}

// Padding the initial state to make all sequences the same length
void CSVDataHandler::padInitialStates(bool end)
{
    size_t max_rows = 0;

    for (const auto &time_vector : _times)
    {
        if (time_vector.size() > max_rows)
        {
            max_rows = time_vector.size();
        }
    }

    auto fillSingleInitialState = [](std::vector<double> &time, std::vector<std::vector<double>> &E,
                                     std::vector<std::vector<double>> &S, /*std::vector<std::vector<double>> &C, */ size_t max_rows, bool &end)
    {
        size_t current_rows = time.size();
        size_t rows_to_fill = max_rows - current_rows;
        if (rows_to_fill > 0)
        {
            time.insert(end ? time.end() : time.begin(), rows_to_fill, -1);
            for (auto &e_col : E)
            {
                e_col.insert(end ? e_col.end() : e_col.begin(), rows_to_fill, 0.0);
            }
            for (auto &s_col : S)
            {
                s_col.insert(end ? s_col.end() : s_col.begin(), rows_to_fill, 0.0);
            }
            // for (auto &c_col : C)
            // {
            //     c_col.insert(end ? c_col.end() : c_col.begin(), rows_to_fill, 0.0);
            // }
        }
    };

    for (size_t i = 0; i < _times.size(); ++i)
    {
        fillSingleInitialState(_times[i], _Es[i], _Ss[i], /*_Cs[i],*/ max_rows, end);
    }
}

// Clear all stored data
void CSVDataHandler::clear()
{
    _times.clear();
    _Es.clear();
    _Ss.clear();
    // _Cs.clear();

    _abs_max_E = 0;
    _abs_max_S = 0;
    // _abs_max_C = 0;
}

// Print all stored data
void CSVDataHandler::print(std::ostream &os)
{
    os << "Absolute maximum E: " << _abs_max_E << std::endl;
    os << "Absolute maximum S: " << _abs_max_S << std::endl;
    // os << "Absolute maximum C: " << _abs_max_C << std::endl;
    os << "Data from all CSV files:" << std::endl;
    for (size_t i = 0; i < _times.size(); ++i)
    {
        os << "File " << i + 1 << ":" << std::endl;

        os << "Time: ";
        for (const auto &t : _times[i])
        {
            os << t << " ";
        }
        os << std::endl;

        os << "E: ";
        for (const auto &E : _Es[i])
        {
            for (const auto &val : E)
            {
                os << val << " ";
            }
            os << std::endl;
        }

        os << "S: ";
        for (const auto &S : _Ss[i])
        {
            for (const auto &val : S)
            {
                os << val << " ";
            }
            os << std::endl;
        }

        // os << "C: ";
        // for (const auto &C : _Cs[i])
        // {
        //     for (const auto &val : C)
        //     {
        //         os << val << " ";
        //     }
        //     os << std::endl;
        // }
        os << std::endl;
    }
}

// return {batch, seq}
torch::Tensor
CSVDataHandler::vectorToTensor(const std::vector<std::vector<double>> &data)
{
    std::vector<double> flattened;
    for (const auto &row : data)
    {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return torch::from_blob(flattened.data(), {static_cast<long>(data.size()), static_cast<long>(data[0].size())}, torch::kFloat64).clone();
}

// return {batch, seq, dim}
torch::Tensor CSVDataHandler::vectorToTensor(const std::vector<std::vector<std::vector<double>>> &data)
{
    size_t rows = data.size();
    size_t cols = data[0].size();
    size_t depth = data[0][0].size();

    std::vector<double> flattened;
    for (const auto &matrix : data)
    {
        for (const auto &row : matrix)
        {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }
    }

    torch::Tensor t = torch::from_blob(flattened.data(), {static_cast<long>(rows), static_cast<long>(cols), static_cast<long>(depth)}, torch::kFloat64).clone();

    // {batch, dim, seq} -> {batch, seq, dim}
    t.transpose_(1, 2);
    return t;
}