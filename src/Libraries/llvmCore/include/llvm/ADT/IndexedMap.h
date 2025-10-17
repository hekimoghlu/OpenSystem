/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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

//===- llvm/ADT/IndexedMap.h - An index map implementation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an indexed map. The index map template takes two
// types. The first is the mapped type and the second is a functor
// that maps its argument to a size_t. On instantiation a "null" value
// can be provided to be used as a "does not exist" indicator in the
// map. A member function grow() is provided that given the value of
// the maximally indexed key (the argument of the functor) makes sure
// the map has enough space for it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_INDEXEDMAP_H
#define LLVM_ADT_INDEXEDMAP_H

#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <functional>
#include <vector>

namespace llvm {

template <typename T, typename ToIndexT = llvm::identity<unsigned> >
  class IndexedMap {
    typedef typename ToIndexT::argument_type IndexT;
    typedef std::vector<T> StorageT;
    StorageT storage_;
    T nullVal_;
    ToIndexT toIndex_;

  public:
    IndexedMap() : nullVal_(T()) { }

    explicit IndexedMap(const T& val) : nullVal_(val) { }

    typename StorageT::reference operator[](IndexT n) {
      assert(toIndex_(n) < storage_.size() && "index out of bounds!");
      return storage_[toIndex_(n)];
    }

    typename StorageT::const_reference operator[](IndexT n) const {
      assert(toIndex_(n) < storage_.size() && "index out of bounds!");
      return storage_[toIndex_(n)];
    }

    void reserve(typename StorageT::size_type s) {
      storage_.reserve(s);
    }

    void resize(typename StorageT::size_type s) {
      storage_.resize(s, nullVal_);
    }

    void clear() {
      storage_.clear();
    }

    void grow(IndexT n) {
      unsigned NewSize = toIndex_(n) + 1;
      if (NewSize > storage_.size())
        resize(NewSize);
    }

    bool inBounds(IndexT n) const {
      return toIndex_(n) < storage_.size();
    }

    typename StorageT::size_type size() const {
      return storage_.size();
    }
  };

} // End llvm namespace

#endif
