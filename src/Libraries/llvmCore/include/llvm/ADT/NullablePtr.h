/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

//===- llvm/ADT/NullablePtr.h - A pointer that allows null ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines and implements the NullablePtr class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_NULLABLE_PTR_H
#define LLVM_ADT_NULLABLE_PTR_H

#include <cassert>
#include <cstddef>

namespace llvm {
/// NullablePtr pointer wrapper - NullablePtr is used for APIs where a
/// potentially-null pointer gets passed around that must be explicitly handled
/// in lots of places.  By putting a wrapper around the null pointer, it makes
/// it more likely that the null pointer case will be handled correctly.
template<class T>
class NullablePtr {
  T *Ptr;
public:
  NullablePtr(T *P = 0) : Ptr(P) {}
  
  bool isNull() const { return Ptr == 0; }
  bool isNonNull() const { return Ptr != 0; }

  /// get - Return the pointer if it is non-null.
  const T *get() const {
    assert(Ptr && "Pointer wasn't checked for null!");
    return Ptr;
  }

  /// get - Return the pointer if it is non-null.
  T *get() {
    assert(Ptr && "Pointer wasn't checked for null!");
    return Ptr;
  }
  
  T *getPtrOrNull() { return Ptr; }
  const T *getPtrOrNull() const { return Ptr; }
};
  
} // end namespace llvm

#endif
