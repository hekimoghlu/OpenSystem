/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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

#ifndef TEST_INTEROP_CXX_STDLIB_INPUTS_CUSTOM_COLLECTION_H
#define TEST_INTEROP_CXX_STDLIB_INPUTS_CUSTOM_COLLECTION_H

#include "custom-iterator.h"
#include <iterator>

struct SimpleCollectionNoSubscript {
private:
  int x[5] = {1, 2, 3, 4, 5};

public:
  using iterator = ConstRACIterator;

  iterator begin() const { return iterator(x); }
  iterator end() const { return iterator(x + 5); }
};

struct SimpleCollectionReadOnly {
private:
  int x[5] = {1, 2, 3, 4, 5};

public:
  using iterator = ConstRACIteratorRefPlusEq;

  iterator begin() const { return iterator(x); }
  iterator end() const { return iterator(x + 5); }

  const int& operator[](int index) const { return x[index]; }
};

struct SimpleCollectionReadWrite {
private:
  int x[5] = {1, 2, 3, 4, 5};

public:
  using const_iterator = ConstRACIterator;
  using iterator = MutableRACIterator;

  const_iterator begin() const { return const_iterator(x); }
  const_iterator end() const { return const_iterator(x + 5); }
  iterator begin() { return iterator(x); }
  iterator end() { return iterator(x + 5); }

  const int &operator[](int index) const { return x[index]; }
  int &operator[](int index) { return x[index]; }
};

template <typename T>
struct HasInheritedTemplatedConstRACIterator {
public:
  typedef InheritedTemplatedConstRACIterator<int> iterator;

private:
  iterator b = iterator(1);
  iterator e = iterator(6);

public:
  iterator begin() const { return b; }
  iterator end() const { return e; }
};

typedef HasInheritedTemplatedConstRACIterator<int>
    HasInheritedTemplatedConstRACIteratorInt;

template <typename T>
struct HasInheritedTemplatedConstRACIteratorOutOfLineOps {
public:
  typedef InheritedTemplatedConstRACIteratorOutOfLineOps<int> iterator;

private:
  iterator b = iterator(1);
  iterator e = iterator(4);

public:
  iterator begin() const { return b; }
  iterator end() const { return e; }
};

typedef HasInheritedTemplatedConstRACIteratorOutOfLineOps<int>
    HasInheritedTemplatedConstRACIteratorOutOfLineOpsInt;

#endif // TEST_INTEROP_CXX_STDLIB_INPUTS_CUSTOM_COLLECTION_H
