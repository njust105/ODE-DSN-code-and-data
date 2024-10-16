#pragma once
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <rapidcsv.h>

namespace Utils
{
    class MultiOstream
    {
    public:
        MultiOstream() {}
        MultiOstream(std::initializer_list<std::ostream *> streams) : _streams(streams) {}

        template <typename T>
        MultiOstream &operator<<(const T &value)
        {
            for (auto stream : _streams)
            {
                if (stream)
                {
                    (*stream) << value;
                }
            }
            return *this;
        }

        // Overload for manipulators like std::endl
        MultiOstream &operator<<(std::ostream &(*manip)(std::ostream &))
        {
            for (auto stream : _streams)
            {
                if (stream)
                {
                    (*stream) << manip;
                }
            }
            return *this;
        }

        void setStreams(std::initializer_list<std::ostream *> streams)
        {
            _streams = streams;
        }

        void addStream(std::ostream *stream)
        {
            if (stream)
                _streams.push_back(stream);
        }

        void flush()
        {
            for (auto stream : _streams)
            {
                if (stream)
                {
                    stream->flush();
                }
            }
        }

        void clearStreams()
        {
            _streams.clear();
        }

    protected:
        std::vector<std::ostream *> _streams;
    };

} // namespace Utils
