#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <tuple>
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <initializer_list>

#include "indices.h"
#include "parallel.h"
#include "pystring.h"
#include "levenshtein.h"


namespace py = pybind11;
using namespace pybind11::literals;

template <typename dist_t>
class Levenshtein {
public:
    static const char * const name;
    static const std::string dtype;

    explicit Levenshtein(size_t n_threads) : n_threads(n_threads) {}

    auto distance(const py::object &str1, const py::object &str2) const {
        return pairwise(py::make_tuple(str1), py::make_tuple(str2)).at(0, 0);
    }

    py::array_t<dist_t> pairwise(const py::iterable &iterable) const {
        const auto strings = readStrings({iterable})[0];
        const auto N = strings.size();
        auto *distvec = new dist_t[N * N];
        for (size_t k = 0; k < N; ++k)
            distvec[k * (N + 1)] = 0;
        UpperTriangleIndexTranslator indmaker(N);
        {
            py::gil_scoped_release nogil;
            parallel_for(0, indmaker.maxind, [&](size_t k, size_t thread_id) {
                auto ij = indmaker.toRowCol(k);
                auto d = dist(strings[ij.first], strings[ij.second]);
                distvec[ij.first * N + ij.second] = d;
                distvec[ij.second * N + ij.first] = d;
            }, n_threads);
        }
        return make_array(distvec, {N, N});
    }

    py::array_t<dist_t> pairwise(const py::iterable &iterable1, const py::iterable &iterable2) const {
        const auto strings = readStrings({iterable1, iterable2});
        const auto n_rows = strings[0].size(), n_cols = strings[1].size();
        RectangleIndexTranslator indmaker(n_rows, n_cols);
        auto *distvec = new dist_t[indmaker.maxind];
        {
            py::gil_scoped_release nogil;
            parallel_for(0, indmaker.maxind, [&](size_t k, size_t thread_id) {
                auto ij = indmaker.toRowCol(k);
                distvec[ij.first * n_cols + ij.second] = dist(strings[0][ij.first], strings[1][ij.second]);
            }, n_threads);
        }
        return make_array(distvec, {n_rows, n_cols});
    }

    decltype(auto) sparse_pairwise(const py::iterable &iterable) const {
        const auto strings = readStrings({iterable})[0];
        return _sparse_pairwise(strings, strings, UpperTriangleIndexTranslator(strings.size()));
    }

    decltype(auto) sparse_pairwise(const py::iterable &iterable1, const py::iterable &iterable2) const {
        const auto strings = readStrings({iterable1, iterable2});
        return _sparse_pairwise(strings[0], strings[1],
                                RectangleIndexTranslator(strings[0].size(), strings[1].size()));
    }

    decltype(auto) sparse_pairwise(const py::iterable &iterable, const py::array_t<size_t> &indices) const {
        const auto strings = readStrings({iterable})[0];
        return _sparse_pairwise(strings, strings,
                                ListedIndexTranslator(indices, strings.size(), strings.size()));
    }

    decltype(auto) sparse_pairwise(const py::iterable &iterable1, const py::iterable &iterable2,
                         const py::array_t<size_t> &indices) const {
        const auto strings = readStrings({iterable1, iterable2});
        return _sparse_pairwise(
                strings[0], strings[1],
                ListedIndexTranslator(indices, strings[0].size(), strings[1].size()));
    }

    std::tuple<size_t, double, double, dist_t> medoid(const py::iterable &iterable) const {
        const auto strings = readStrings({iterable})[0];
        const auto N = strings.size();
        if (N == 0)
            throw py::value_error("cannot calculate medoid for an empty group of strings");
        UpperTriangleIndexTranslator indmaker(N);
        std::vector<dist_t> distvec(indmaker.maxind);
        {
            py::gil_scoped_release nogil;
            parallel_for(0, distvec.size(), [&](size_t k, size_t thread_id) {
                auto ij = indmaker.toRowCol(k);
                distvec[k] = dist(strings[ij.first], strings[ij.second]);
            }, n_threads);
        }
        std::vector<dist_t> sum(N, 0);
        for (size_t k = 0; k < distvec.size(); ++k) {
            auto ij = indmaker.toRowCol(k);
            sum[ij.first] += distvec[k];
            sum[ij.second] += distvec[k];
        }
        auto min = std::min_element(sum.begin(), sum.end());
        auto medoid = static_cast<size_t>(min - sum.begin());
        auto mean = static_cast<double>(*min) / N;
        dist_t sum2 = 0, max_dist = 0;
        for (size_t other = 0; other < N; ++other) {
            if (other == medoid)
                continue;
            auto d = distvec[indmaker.toFlat(other, medoid)];
            sum2 += d * d;
            if (d > max_dist)
                max_dist = d;
        }
        return {medoid, mean, std::sqrt(static_cast<double>(sum2) / N - mean * mean), max_dist};
    }

private:
    static const py::object csr_matrix;
    const size_t n_threads;

    // The two strings are assumed to be of the same kind
    auto dist(const PyStringWrap &a, const PyStringWrap &b) const {
        dist_t d;
        auto pa = a.ptr(), pb = b.ptr();
        auto la = a.len(), lb = b.len();
        switch (a.kind()) {
            case PyUnicode_WCHAR_KIND:
                levendist(reinterpret_cast<const char *>(pa), la, reinterpret_cast<const char *>(pb), lb, d);
                break;
            case PyUnicode_1BYTE_KIND:
                levendist(reinterpret_cast<const Py_UCS1 *>(pa), la, reinterpret_cast<const Py_UCS1 *>(pb), lb, d);
                break;
            case PyUnicode_2BYTE_KIND:
                levendist(reinterpret_cast<const Py_UCS2 *>(pa), la, reinterpret_cast<const Py_UCS2 *>(pb), lb, d);
                break;
            case PyUnicode_4BYTE_KIND:
                levendist(reinterpret_cast<const Py_UCS4 *>(pa), la, reinterpret_cast<const Py_UCS4 *>(pb), lb, d);
                break;
            default:
                throw std::runtime_error("unexpected unicode kind");
        }
        return d;
    }

    auto readStrings(std::initializer_list<py::iterable> iterables) const {
        std::vector<std::vector<PyStringWrap>> strings(iterables.size());
        bool convert_kind = false;
        std::optional<PyUnicode_Kind> seen_kind;
        std::transform(iterables.begin(), iterables.end(), strings.begin(), [&](const py::iterable &iterable) {
            std::vector<PyStringWrap> strvec;
            for (auto it = py::iter(iterable); it != py::iterator::sentinel(); ++it) {
                strvec.emplace_back(*it);
                const auto kind = strvec.back().kind();
                if (!seen_kind)
                    seen_kind = kind;
                else if (seen_kind.value() != kind) {
                    convert_kind = true;
                    if (seen_kind.value() == PyUnicode_WCHAR_KIND || kind == PyUnicode_WCHAR_KIND)
                        throw py::value_error("cannot mix bytes/bytearray objects and unicode strings");
                }
            }
            return strvec;
        });
        if (convert_kind)
            for (auto &strvec : strings)
                for (auto &s : strvec)
                    s.toUnicode4();
        return strings;
    }

    template <typename dtype>
    static auto make_array(dtype *data, std::initializer_list<size_t> shape) {
        return py::array_t<dtype>{
            shape, data, py::capsule(data, [](void *p) { delete[] reinterpret_cast<dtype *>(p); })
        };
    }

    py::object _sparse_pairwise(const std::vector<PyStringWrap> &strings1, const std::vector<PyStringWrap> &strings2,
                                const IndexTranslator &indmaker) const {
        const auto N = indmaker.maxind;
        auto *distvec = new dist_t[N];
        auto *row = new size_t[N];
        auto *col = new size_t[N];
        {
            py::gil_scoped_release nogil;
            parallel_for(0, N, [&](size_t k, size_t thread_id) {
                auto ij = indmaker.toRowCol(k);
                distvec[k] = dist(strings1[ij.first], strings2[ij.second]);
                row[k] = ij.first;
                col[k] = ij.second;
            }, n_threads);
        }
        return csr_matrix(
                py::make_tuple(
                        make_array(distvec, {N}),
                        py::make_tuple(make_array(row, {N}), make_array(col, {N}))
                ),
                py::make_tuple(strings1.size(), strings2.size())
        );
    }
};

template <>
const char * const Levenshtein<unsigned>::name = "Levenshtein<normalize=False>";

template <>
const char * const Levenshtein<double>::name = "Levenshtein<normalize=True>";

template <>
const std::string Levenshtein<unsigned>::dtype("int");

template <>
const std::string Levenshtein<double>::dtype("float");

template <typename dist_t>
const py::object Levenshtein<dist_t>::csr_matrix = py::module_::import("scipy.sparse").attr("csr_matrix");


template <typename dist_t>
void bind(const py::module &module) {
    using Leven = Levenshtein<dist_t>;
    py::class_<Leven>(module, Leven::name)
        .def("distance", &Leven::distance, "str1"_a, "str2"_a,
             std::string("Calculate the distance between two strings.\n\n"
             "Parameters\n"
             "----------\n"
             "str1, str2 : str / bytes / bytearray\n"
             "    The strings to compare.\n"
             "    Must either both be of type str or a mix of byte and bytearray.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "distance : ").append(Leven::dtype).append("\n"
             "    The Levenshtein distance between str1 and str2.\n").c_str()
        )

        .def("pairwise", py::overload_cast<const py::iterable &>(&Leven::pairwise, py::const_),
             "strings"_a,
             std::string("Calculate the pairwise distances between some strings.\n\n"
             "Parameters\n"
             "----------\n"
             "strings : iterable of str / bytes / bytearray, length n_strings\n"
             "    The strings to compare.\n"
             "    Must either all be of type str or a mix of byte and bytearray.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "pairwise_distances : symmetric numpy array of ")
             .append(Leven::dtype).append("s, shape [n_strings, n_strings]\n"
             "    The pairwise distances between the strings.\n").c_str()
        )
        .def("pairwise",
             py::overload_cast<const py::iterable &, const py::iterable &>(&Leven::pairwise, py::const_),
            "strings1"_a, "strings2"_a,
             std::string("Calculate the pairwise distances between two groups of strings.\n\n"
             "Parameters\n"
             "----------\n"
             "strings1 : iterable of str / bytes / bytearray, length N\n"
             "    The first group of strings.\n"
             "\n"
             "strings2 : iterable of str / bytes / bytearray, length M\n"
             "    The second group of strings.\n"
             "\n"
             "* Note: The strings in both groups must either all be of type str or a mix of byte and bytearray.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "pairwise_distances : numpy array of ").append(Leven::dtype).append("s, shape [N, M]\n"
             "    The pairwise distances between `strings1` and `strings2`.\n").c_str()
        )

        .def("sparse_pairwise",
            [](const Leven &self, const py::iterable &strings1, const py::object &strings2,
               const py::object &indices) {
                if (strings2.is_none())
                    return indices.is_none() ?
                           self.sparse_pairwise(strings1) :
                           self.sparse_pairwise(strings1, py::array_t<size_t>(indices));
                auto str2 = strings2.cast<py::iterable>();
                return indices.is_none() ?
                       self.sparse_pairwise(strings1, str2) :
                       self.sparse_pairwise(strings1, str2, py::array_t<size_t>(indices));
            }, "strings1"_a, "strings2"_a=py::none(), "indices"_a=py::none(),
             std::string("Calculate a sparse pairwise distances matrix.\n\n"
             "Parameters\n"
             "----------\n"
             "strings1 : iterable of str / bytes / bytearray, length N\n"
             "    The first group of strings.\n"
             "\n"
             "strings2 : iterable of str / bytes / bytearray, length M, optional\n"
             "    The second group of strings.\n"
             "    If not provided, the pairwise distances matrix of `strings1` with itself will be calculated.\n"
             "\n"
             "* Note: The strings in both groups must either all be of type str or a mix of byte and bytearray.\n"
             "\n"
             "indices : array_like with two columns, optional\n"
             "    Pairs of indices to calculate.\n"
             "    If not provided, all pairs will be calculated.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "pairwise_distances : scipy.sparse.csr_matrix of ").append(Leven::dtype).append("s, shape [N, M]\n"
             "    The pairwise distances.\n"
             "    If neither `strings2` nor `indices` are provided, only the upper triangle will be filled.\n"
             ).c_str()
        )

        .def("medoid", &Leven::medoid, "strings"_a,
             std::string("Calculate the medoid and other statistics of some strings.\n\n"
             "Parameters\n"
             "----------\n"
             "strings : iterable of str / bytes / bytearray, length n_strings\n"
             "    The strings to compare.\n"
             "    Must either all be of type str or a mix of byte and bytearray.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "statistics : tuple, length 4\n"
             "    medoid : int\n"
             "        The index of the input string which is the medoid.\n"
             "    avg_distance : float\n"
             "        The average distance of all strings to the medoid.\n"
             "    std_distance : float\n"
             "        The standard deviation of the distances to the medoid.\n"
             "    max_distance : ").append(Leven::dtype).append("\n"
             "        The maximum distance to the medoid.\n"
             ).c_str()
        );
}

PYBIND11_MODULE(leven, module) {
    module.def(
        "Levenshtein", [](bool normalize=false, size_t n_threads=0) {
            py::object obj;
            if (normalize)
                obj = py::cast(new Levenshtein<double>(n_threads), py::return_value_policy::take_ownership);
            else
                obj = py::cast(new Levenshtein<unsigned>(n_threads), py::return_value_policy::take_ownership);
            return obj;
        },
        "normalize"_a=false, "n_threads"_a=0,
        "Create a Levenshtein distance calculator.\n\n"
        "Parameters\n"
        "----------\n"
        "normalize : bool, default False\n"
        "    Normalize distance by dividing by the length of the longer of the two strings.\n"
        "\n"
        "n_threads : int, default 0\n"
        "    Number of threads to use for distance calculations.\n"
        "    If 0, use number of cores.\n"
    );

    bind<double>(module);
    bind<unsigned>(module);
}
