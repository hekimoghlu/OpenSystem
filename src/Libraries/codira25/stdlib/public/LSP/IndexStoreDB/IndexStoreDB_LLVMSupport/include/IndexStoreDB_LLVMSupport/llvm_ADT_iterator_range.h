/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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

//===- iterator_range.h - A range adaptor for iterators ---------*- C++ -*-===//
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
/// \file
/// This provides a very simple, boring adaptor for a begin and end iterator
/// into a range type. This should be used to build range views that work well
/// with range based for loops and range based constructors.
///
/// Note that code here follows more standards-based coding conventions as it
/// is mirroring proposed interfaces for standardization.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ITERATOR_RANGE_H
#define LLVM_ADT_ITERATOR_RANGE_H

#include <IndexStoreDB_LLVMSupport/toolchain_Config_indexstoredb-prefix.h>

#include <iterator>
#include <utility>

namespace toolchain {

/// A range adaptor for a pair of iterators.
///
/// This just wraps two iterators into a range-compatible interface. Nothing
/// fancy at all.
template <typename IteratorT>
class iterator_range {
  IteratorT begin_iterator, end_iterator;

public:
  //TODO: Add SFINAE to test that the Container's iterators match the range's
  //      iterators.
  template <typename Container>
  iterator_range(Container &&c)
  //TODO: Consider ADL/non-member begin/end calls.
      : begin_iterator(c.begin()), end_iterator(c.end()) {}
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator(std::move(begin_iterator)),
        end_iterator(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator; }
  IteratorT end() const { return end_iterator; }
};

/// Convenience function for iterating over sub-ranges.
///
/// This provides a bit of syntactic sugar to make using sub-ranges
/// in for loops a bit easier. Analogous to std::make_pair().
template <class T> iterator_range<T> make_range(T x, T y) {
  return iterator_range<T>(std::move(x), std::move(y));
}

template <typename T> iterator_range<T> make_range(std::pair<T, T> p) {
  return iterator_range<T>(std::move(p.first), std::move(p.second));
}

template <typename T>
iterator_range<decltype(adl_begin(std::declval<T>()))> drop_begin(T &&t,
                                                                  int n) {
  return make_range(std::next(adl_begin(t), n), adl_end(t));
}
}

#endif
