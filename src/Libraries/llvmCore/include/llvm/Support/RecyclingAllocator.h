/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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

//==- llvm/Support/RecyclingAllocator.h - Recycling Allocator ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RecyclingAllocator class.  See the doxygen comment for
// RecyclingAllocator for more details on the implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RECYCLINGALLOCATOR_H
#define LLVM_SUPPORT_RECYCLINGALLOCATOR_H

#include "llvm/Support/Recycler.h"

namespace llvm {

/// RecyclingAllocator - This class wraps an Allocator, adding the
/// functionality of recycling deleted objects.
///
template<class AllocatorType, class T,
         size_t Size = sizeof(T), size_t Align = AlignOf<T>::Alignment>
class RecyclingAllocator {
private:
  /// Base - Implementation details.
  ///
  Recycler<T, Size, Align> Base;

  /// Allocator - The wrapped allocator.
  ///
  AllocatorType Allocator;

public:
  ~RecyclingAllocator() { Base.clear(Allocator); }

  /// Allocate - Return a pointer to storage for an object of type
  /// SubClass. The storage may be either newly allocated or recycled.
  ///
  template<class SubClass>
  SubClass *Allocate() { return Base.template Allocate<SubClass>(Allocator); }

  T *Allocate() { return Base.Allocate(Allocator); }

  /// Deallocate - Release storage for the pointed-to object. The
  /// storage will be kept track of and may be recycled.
  ///
  template<class SubClass>
  void Deallocate(SubClass* E) { return Base.Deallocate(Allocator, E); }

  void PrintStats() { Base.PrintStats(); }
};

}

template<class AllocatorType, class T, size_t Size, size_t Align>
inline void *operator new(size_t,
                          llvm::RecyclingAllocator<AllocatorType,
                                                   T, Size, Align> &Allocator) {
  return Allocator.Allocate();
}

template<class AllocatorType, class T, size_t Size, size_t Align>
inline void operator delete(void *E,
                            llvm::RecyclingAllocator<AllocatorType,
                                                     T, Size, Align> &A) {
  A.Deallocate(E);
}

#endif
