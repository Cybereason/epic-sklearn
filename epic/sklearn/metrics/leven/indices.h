#pragma once

#include <Python.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <utility>
#include <cstdlib>


namespace py = pybind11;

class IndexTranslator {
public:
    using indpair = std::pair<size_t, size_t>;
    explicit IndexTranslator(size_t maxind) : maxind(maxind) {}
    virtual indpair toRowCol(size_t flat) const = 0;
    const size_t maxind;
};


class UpperTriangleIndexTranslator : public IndexTranslator {
public:
    explicit UpperTriangleIndexTranslator(size_t size)
    : IndexTranslator(size * (size - 1) / 2), size(size), size_m_half(size - 0.5),
    size_m_half2(size_m_half * size_m_half) {}

    indpair toRowCol(size_t flat) const override {
        auto row = static_cast<size_t>(size_m_half - std::sqrt(size_m_half2 - 2 * flat));
        size_t col = row * (row + 3) / 2 + flat - row * size + 1;
        return {row, col};
    }

    inline auto toFlat(size_t row, size_t col) const {
        if (row > col)
            std::swap(row, col);
        return row * size - row * (row + 3) / 2 + col - 1;
    }

private:
    const size_t size;
    const double size_m_half, size_m_half2;
};


class RectangleIndexTranslator : public IndexTranslator {
public:
    explicit RectangleIndexTranslator(size_t n_rows, size_t n_cols)
    : IndexTranslator(n_rows * n_cols), n_cols(static_cast<int>(n_cols)) {}

    indpair toRowCol(size_t flat) const override {
        auto divmod = std::div(static_cast<int>(flat), n_cols);
        return {divmod.quot, divmod.rem};
    }

private:
    const int n_cols;
};


class ListedIndexTranslator : public IndexTranslator {
public:
    explicit ListedIndexTranslator(const py::array_t<size_t> &indices, size_t n_rows, size_t n_cols)
    : IndexTranslator(indices.shape(0)), _indices(indices.unchecked<2>()) {
        if (indices.ndim() != 2)
            throw py::value_error("`indices` must be 2-dimensional");
        if (indices.shape(1) != 2)
            throw py::value_error("`indices` must have 2 columns");
        for (size_t i = 0; i < maxind; ++i)
            if (_indices(i, 0) >= n_rows || _indices(i, 1) >= n_cols)
                throw py::value_error("`indices` contains illegal values");
    }

    indpair toRowCol(size_t flat) const override {
        return {_indices(flat, 0), _indices(flat, 1)};
    }

private:
    py::detail::unchecked_reference<size_t, 2> _indices;
};
