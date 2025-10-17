/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//===-- language/Compability/Parser/char-block.h -----------------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// Author: Tunjay Akbarli
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_COMPABILITY_PARSER_CHAR_BLOCK_H_
#define LANGUAGE_COMPABILITY_PARSER_CHAR_BLOCK_H_

// Describes a contiguous block of characters; does not own their storage.

#include "language/Compability/Common/interval.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iosfwd>
#include <string>
#include <utility>

namespace toolchain {
class raw_ostream;
}

namespace language::Compability::parser {

class CharBlock {
public:
  constexpr CharBlock() {}
  constexpr CharBlock(const char *x, std::size_t n = 1) : interval_{x, n} {}
  constexpr CharBlock(const char *b, const char *ep1)
      : interval_{b, static_cast<std::size_t>(ep1 - b)} {}
  CharBlock(const std::string &s) : interval_{s.data(), s.size()} {}
  constexpr CharBlock(const CharBlock &) = default;
  constexpr CharBlock(CharBlock &&) = default;
  constexpr CharBlock &operator=(const CharBlock &) = default;
  constexpr CharBlock &operator=(CharBlock &&) = default;

  constexpr bool empty() const { return interval_.empty(); }
  constexpr std::size_t size() const { return interval_.size(); }
  constexpr const char *begin() const { return interval_.start(); }
  constexpr const char *end() const {
    return interval_.start() + interval_.size();
  }
  constexpr const char &operator[](std::size_t j) const {
    return interval_.start()[j];
  }
  constexpr const char &front() const { return (*this)[0]; }
  constexpr const char &back() const { return (*this)[size() - 1]; }

  bool Contains(const CharBlock &that) const {
    return interval_.Contains(that.interval_);
  }

  void ExtendToCover(const CharBlock &that) {
    interval_.ExtendToCover(that.interval_);
  }

  // Returns the block's first non-blank character, if it has
  // one; otherwise ' '.
  char FirstNonBlank() const {
    for (char ch : *this) {
      if (ch != ' ' && ch != '\t') {
        return ch;
      }
    }
    return ' '; // non no-blank character
  }

  // Returns the block's only non-blank character, if it has
  // exactly one non-blank character; otherwise ' '.
  char OnlyNonBlank() const {
    char result{' '};
    for (char ch : *this) {
      if (ch != ' ' && ch != '\t') {
        if (result == ' ') {
          result = ch;
        } else {
          return ' ';
        }
      }
    }
    return result;
  }

  std::size_t CountLeadingBlanks() const {
    std::size_t n{size()};
    std::size_t j{0};
    for (; j < n; ++j) {
      char ch{(*this)[j]};
      if (ch != ' ' && ch != '\t') {
        break;
      }
    }
    return j;
  }

  bool IsBlank() const { return FirstNonBlank() == ' '; }

  std::string ToString() const {
    return std::string{interval_.start(), interval_.size()};
  }

  // Convert to string, stopping early at any embedded '\0'.
  std::string NULTerminatedToString() const {
    return std::string{interval_.start(),
        /*not in std::*/ strnlen(interval_.start(), interval_.size())};
  }

  bool operator<(const CharBlock &that) const { return Compare(that) < 0; }
  bool operator<=(const CharBlock &that) const { return Compare(that) <= 0; }
  bool operator==(const CharBlock &that) const { return Compare(that) == 0; }
  bool operator!=(const CharBlock &that) const { return Compare(that) != 0; }
  bool operator>=(const CharBlock &that) const { return Compare(that) >= 0; }
  bool operator>(const CharBlock &that) const { return Compare(that) > 0; }

  bool operator<(const char *that) const { return Compare(that) < 0; }
  bool operator<=(const char *that) const { return Compare(that) <= 0; }
  bool operator==(const char *that) const { return Compare(that) == 0; }
  bool operator!=(const char *that) const { return Compare(that) != 0; }
  bool operator>=(const char *that) const { return Compare(that) >= 0; }
  bool operator>(const char *that) const { return Compare(that) > 0; }

  friend bool operator<(const char *, const CharBlock &);
  friend bool operator<=(const char *, const CharBlock &);
  friend bool operator==(const char *, const CharBlock &);
  friend bool operator!=(const char *, const CharBlock &);
  friend bool operator>=(const char *, const CharBlock &);
  friend bool operator>(const char *, const CharBlock &);

private:
  int Compare(const CharBlock &that) const {
    // "memcmp" in glibc has "nonnull" attributes on the input pointers.
    // Avoid passing null pointers, since it would result in an undefined
    // behavior.
    if (size() == 0) {
      return that.size() == 0 ? 0 : -1;
    } else if (that.size() == 0) {
      return 1;
    } else {
      std::size_t bytes{std::min(size(), that.size())};
      int cmp{std::memcmp(static_cast<const void *>(begin()),
          static_cast<const void *>(that.begin()), bytes)};
      if (cmp != 0) {
        return cmp;
      } else {
        return size() < that.size() ? -1 : size() > that.size();
      }
    }
  }

  int Compare(const char *that) const {
    std::size_t bytes{size()};
    // strncmp is undefined if either pointer is null.
    if (!bytes) {
      return that == nullptr ? 0 : -1;
    } else if (!that) {
      return 1;
    } else if (int cmp{std::strncmp(begin(), that, bytes)}) {
      return cmp;
    }
    return that[bytes] == '\0' ? 0 : -1;
  }

  common::Interval<const char *> interval_{nullptr, 0};
};

inline bool operator<(const char *left, const CharBlock &right) {
  return right > left;
}
inline bool operator<=(const char *left, const CharBlock &right) {
  return right >= left;
}
inline bool operator==(const char *left, const CharBlock &right) {
  return right == left;
}
inline bool operator!=(const char *left, const CharBlock &right) {
  return right != left;
}
inline bool operator>=(const char *left, const CharBlock &right) {
  return right <= left;
}
inline bool operator>(const char *left, const CharBlock &right) {
  return right < left;
}

// An alternative comparator based on pointer values; use with care!
struct CharBlockPointerComparator {
  bool operator()(CharBlock x, CharBlock y) const {
    return x.end() < y.begin();
  }
};

toolchain::raw_ostream &operator<<(toolchain::raw_ostream &os, const CharBlock &x);

} // namespace language::Compability::parser

// Specializations to enable std::unordered_map<CharBlock, ...> &c.
template <> struct std::hash<language::Compability::parser::CharBlock> {
  std::size_t operator()(const language::Compability::parser::CharBlock &x) const {
    std::size_t hash{0}, bytes{x.size()};
    for (std::size_t j{0}; j < bytes; ++j) {
      hash = (hash * 31) ^ x[j];
    }
    return hash;
  }
};
#endif // FORTRAN_PARSER_CHAR_BLOCK_H_
