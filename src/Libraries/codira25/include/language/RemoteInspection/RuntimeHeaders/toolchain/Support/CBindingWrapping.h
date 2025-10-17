/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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

//===- toolchain/Support/CBindingWrapping.h - C Interface Wrapping ---*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
// This file declares the wrapping macros for the C interface.
//
//===----------------------------------------------------------------------===//

#ifndef TOOLCHAIN_SUPPORT_CBINDINGWRAPPING_H
#define TOOLCHAIN_SUPPORT_CBINDINGWRAPPING_H

#include "toolchain-c/Types.h"
#include "toolchain/Support/Casting.h"

#define DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)     \
  inline ty *unwrap(ref P) {                            \
    return reinterpret_cast<ty*>(P);                    \
  }                                                     \
                                                        \
  inline ref wrap(const ty *P) {                        \
    return reinterpret_cast<ref>(const_cast<ty*>(P));   \
  }

#define DEFINE_ISA_CONVERSION_FUNCTIONS(ty, ref)        \
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)           \
                                                        \
  template<typename T>                                  \
  inline T *unwrap(ref P) {                             \
    return cast<T>(unwrap(P));                          \
  }

#define DEFINE_STDCXX_CONVERSION_FUNCTIONS(ty, ref)     \
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)           \
                                                        \
  template<typename T>                                  \
  inline T *unwrap(ref P) {                             \
    T *Q = (T*)unwrap(P);                               \
    assert(Q && "Invalid cast!");                       \
    return Q;                                           \
  }

#endif
