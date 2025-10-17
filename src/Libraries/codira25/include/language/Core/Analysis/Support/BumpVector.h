/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

//===- BumpVector.h - Vector-like ADT that uses bump allocation -*- C++ -*-===//
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
//
//  This file provides BumpVector, a vector-like ADT whose contents are
//  allocated from a BumpPtrAllocator.
//
//===----------------------------------------------------------------------===//

// FIXME: Most of this is copy-and-paste from SmallVector.h.  We can
// refactor this core logic into something common that is shared between
// the two.  The main thing that is different is the allocation strategy.

#ifndef LANGUAGE_CORE_ANALYSIS_SUPPORT_BUMPVECTOR_H
#define LANGUAGE_CORE_ANALYSIS_SUPPORT_BUMPVECTOR_H

#include "toolchain/ADT/PointerIntPair.h"
#include "toolchain/Support/Allocator.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <type_traits>

namespace language::Core {

class BumpVectorContext {
  toolchain::PointerIntPair<toolchain::BumpPtrAllocator*, 1> Alloc;

public:
  /// Construct a new BumpVectorContext that creates a new BumpPtrAllocator
  /// and destroys it when the BumpVectorContext object is destroyed.
  BumpVectorContext() : Alloc(new toolchain::BumpPtrAllocator(), 1) {}

  BumpVectorContext(BumpVectorContext &&Other) : Alloc(Other.Alloc) {
    Other.Alloc.setInt(false);
    Other.Alloc.setPointer(nullptr);
  }

  // The move assignment operator is defined as deleted pending further
  // motivation.
  BumpVectorContext &operator=(BumpVectorContext &&) = delete;

  // The copy constrcutor and copy assignment operator is defined as deleted
  // pending further motivation.
  BumpVectorContext(const BumpVectorContext &) = delete;
  BumpVectorContext &operator=(const BumpVectorContext &) = delete;

  /// Construct a new BumpVectorContext that reuses an existing
  /// BumpPtrAllocator.  This BumpPtrAllocator is not destroyed when the
  /// BumpVectorContext object is destroyed.
  BumpVectorContext(toolchain::BumpPtrAllocator &A) : Alloc(&A, 0) {}

  ~BumpVectorContext() {
    if (Alloc.getInt())
      delete Alloc.getPointer();
  }

  toolchain::BumpPtrAllocator &getAllocator() { return *Alloc.getPointer(); }
};

template<typename T>
class BumpVector {
  T *Begin = nullptr;
  T *End = nullptr;
  T *Capacity = nullptr;

public:
  // Default ctor - Initialize to empty.
  explicit BumpVector(BumpVectorContext &C, unsigned N) {
    reserve(C, N);
  }

  ~BumpVector() {
    if (std::is_class<T>::value) {
      // Destroy the constructed elements in the vector.
      destroy_range(Begin, End);
    }
  }

  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using iterator = T *;
  using const_iterator = const T *;

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = std::reverse_iterator<iterator>;

  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;

  // forward iterator creation methods.
  iterator begin() { return Begin; }
  const_iterator begin() const { return Begin; }
  iterator end() { return End; }
  const_iterator end() const { return End; }

  // reverse iterator creation methods.
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  bool empty() const { return Begin == End; }
  size_type size() const { return End-Begin; }

  reference operator[](unsigned idx) {
    assert(Begin + idx < End);
    return Begin[idx];
  }
  const_reference operator[](unsigned idx) const {
    assert(Begin + idx < End);
    return Begin[idx];
  }

  reference front() {
    return begin()[0];
  }
  const_reference front() const {
    return begin()[0];
  }

  reference back() {
    return end()[-1];
  }
  const_reference back() const {
    return end()[-1];
  }

  void pop_back() {
    --End;
    End->~T();
  }

  T pop_back_val() {
    T Result = back();
    pop_back();
    return Result;
  }

  void clear() {
    if (std::is_class<T>::value) {
      destroy_range(Begin, End);
    }
    End = Begin;
  }

  /// data - Return a pointer to the vector's buffer, even if empty().
  pointer data() {
    return pointer(Begin);
  }

  /// data - Return a pointer to the vector's buffer, even if empty().
  const_pointer data() const {
    return const_pointer(Begin);
  }

  void push_back(const_reference Elt, BumpVectorContext &C) {
    if (End < Capacity) {
    Retry:
      new (End) T(Elt);
      ++End;
      return;
    }
    grow(C);
    goto Retry;
  }

  /// insert - Insert some number of copies of element into a position. Return
  /// iterator to position after last inserted copy.
  iterator insert(iterator I, size_t Cnt, const_reference E,
      BumpVectorContext &C) {
    assert(I >= Begin && I <= End && "Iterator out of bounds.");
    if (End + Cnt <= Capacity) {
    Retry:
      move_range_right(I, End, Cnt);
      construct_range(I, I + Cnt, E);
      End += Cnt;
      return I + Cnt;
    }
    ptrdiff_t D = I - Begin;
    grow(C, size() + Cnt);
    I = Begin + D;
    goto Retry;
  }

  void reserve(BumpVectorContext &C, unsigned N) {
    if (unsigned(Capacity-Begin) < N)
      grow(C, N);
  }

  /// capacity - Return the total number of elements in the currently allocated
  /// buffer.
  size_t capacity() const { return Capacity - Begin; }

private:
  /// grow - double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(BumpVectorContext &C, size_type MinSize = 1);

  void construct_range(T *S, T *E, const T &Elt) {
    for (; S != E; ++S)
      new (S) T(Elt);
  }

  void destroy_range(T *S, T *E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  void move_range_right(T *S, T *E, size_t D) {
    for (T *I = E + D - 1, *IL = S + D - 1; I != IL; --I) {
      --E;
      new (I) T(*E);
      E->~T();
    }
  }
};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T>
void BumpVector<T>::grow(BumpVectorContext &C, size_t MinSize) {
  size_t CurCapacity = Capacity-Begin;
  size_t CurSize = size();
  size_t NewCapacity = 2*CurCapacity;
  if (NewCapacity < MinSize)
    NewCapacity = MinSize;

  // Allocate the memory from the BumpPtrAllocator.
  T *NewElts = C.getAllocator().template Allocate<T>(NewCapacity);

  // Copy the elements over.
  if (Begin != End) {
    if (std::is_class<T>::value) {
      std::uninitialized_copy(Begin, End, NewElts);
      // Destroy the original elements.
      destroy_range(Begin, End);
    } else {
      // Use memcpy for PODs (std::uninitialized_copy optimizes to memmove).
      memcpy(NewElts, Begin, CurSize * sizeof(T));
    }
  }

  // For now, leak 'Begin'.  We can add it back to a freelist in
  // BumpVectorContext.
  Begin = NewElts;
  End = NewElts+CurSize;
  Capacity = Begin+NewCapacity;
}

} // namespace language::Core

#endif // LANGUAGE_CORE_ANALYSIS_SUPPORT_BUMPVECTOR_H
