#pragma once

#include <algorithm>
#include <levenshtein_impl.h>


template<typename char_t>
void levendist(const char_t *str1, size_t len1, const char_t *str2, size_t len2, unsigned &dist)
{
    dist = levenshtein(str1, len1, str2, len2);
}


template<typename char_t>
void levendist(const char_t *str1, size_t len1, const char_t *str2, size_t len2, double &dist)
{
    dist = static_cast<double>(levenshtein(str1, len1, str2, len2));
    auto size = std::max(len1, len2);
    if (size)
        dist /= static_cast<double>(size);
}
