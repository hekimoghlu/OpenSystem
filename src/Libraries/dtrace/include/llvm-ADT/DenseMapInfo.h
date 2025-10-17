/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 6, 2022.
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

//===- llvm/ADT/DenseMapInfo.h - Type traits for DenseMap -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines DenseMapInfo traits for DenseMap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSEMAPINFO_H
#define LLVM_ADT_DENSEMAPINFO_H

#include <memory>

namespace llvm {

template<typename T>
struct DenseMapInfo {
  //static inline T getEmptyKey();
  //static inline T getTombstoneKey();
  //static unsigned getHashValue(const T &Val);
  //static bool isEqual(const T &LHS, const T &RHS);
};

template<>
struct DenseMapInfo<const char *> {
  static inline const char *getEmptyKey() {
    return reinterpret_cast<const char *>(-1);
  }
  static inline const char *getTombstoneKey() {
    return reinterpret_cast<const char *>(-2);
  }
  static unsigned getHashValue(const char *s) {
    return std::__murmur2_or_cityhash<size_t>()(s, strlen(s));
  }
  static bool isEqual(const char *LHS, const char *RHS) {
    if (LHS == RHS)
      return true;
    if (LHS == getEmptyKey() || RHS == getEmptyKey())
      return false;
    if (LHS == getTombstoneKey() || RHS == getTombstoneKey())
      return false;
    return strcmp(LHS, RHS) == 0;
  }
};


} // end namespace llvm

#endif // LLVM_ADT_DENSEMAPINFO_H
