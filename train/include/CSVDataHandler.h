#pragma once

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <rapidcsv.h>
#include <torch/torch.h>

class CSVDataHandler
{
public:
    explicit CSVDataHandler(int dimension);
    CSVDataHandler(const std::string &directoryPath, int dimension, unsigned int min_length = 0, const bool enable_padding = false);
    void readAndStoreData(const std::string &directoryPath);
    void readCSVFile(const std::string &filePath, bool findMax = true);
    void filterShortSeries(unsigned int min_length);
    void findMax();
    void padInitialStates(bool end = true);

    // number of sequences
    inline unsigned int size() { return _times.size(); }

    void clear();
    void print(std::ostream &os = std::cout);

    // size: {batch, seq}
    inline torch::Tensor t() { return vectorToTensor(_times); }
    // size: {batch, seq, 6 or 3}
    inline torch::Tensor E() { return vectorToTensor(_Es); }
    // size: {batch, seq, 6 or 3}
    inline torch::Tensor S() { return vectorToTensor(_Ss); }
    // // size: {batch, seq, 21 or 6}
    // inline torch::Tensor C() { return vectorToTensor(_Cs); }

    // size: {batch, seq, 6 or 3}
    inline torch::Tensor normalized_E() { return E() / _abs_max_E; }
    // size: {batch, seq, 6 or 3}
    inline torch::Tensor normalized_S() { return S() / _abs_max_S; }
    // // size: {batch, seq, 21 or 6}
    // inline torch::Tensor normalized_C() { return C() / _abs_max_C; }

    inline double abs_max_E() { return _abs_max_E; }
    inline double abs_max_S() { return _abs_max_S; }
    // inline double abs_max_C() { return _abs_max_C; }

protected:
    std::vector<std::vector<double>> readMultipleColumns(rapidcsv::Document &doc, const std::vector<std::string> &headers);
    bool checkHeaders(rapidcsv::Document &doc);

    torch::Tensor vectorToTensor(const std::vector<std::vector<double>> &data);
    torch::Tensor vectorToTensor(const std::vector<std::vector<std::vector<double>>> &data);

    int _dimension;
    std::string _time_header;
    std::vector<std::string> _E_headers;
    std::vector<std::string> _S_headers;
    // std::vector<std::string> _C_headers;
    std::vector<std::string> _all_headers;

    // Save the data of all files that have been read.
    std::vector<std::vector<double>> _times;
    std::vector<std::vector<std::vector<double>>> _Es;
    std::vector<std::vector<std::vector<double>>> _Ss;
    // std::vector<std::vector<std::vector<double>>> _Cs;

    double _abs_max_E;
    double _abs_max_S;
    // double _abs_max_C;

    // Save the data of the file that is currently being read.
    std::vector<double> _current_file_time;
    std::vector<std::vector<double>> _current_file_E;
    std::vector<std::vector<double>> _current_file_S;
    // std::vector<std::vector<double>> _current_file_C;
};
