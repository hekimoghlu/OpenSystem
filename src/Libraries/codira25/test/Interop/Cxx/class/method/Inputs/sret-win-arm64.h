/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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

#ifndef TEST_INTEROP_CXX_CLASS_METHOD_SRET_WIN_ARM64_H
#define TEST_INTEROP_CXX_CLASS_METHOD_SRET_WIN_ARM64_H

#include <stdint.h>

namespace toolchain {

template<typename T>
class ArrayRef {
public:
   const T *Data = nullptr;
   size_t Length = 0;
};

} // namespace toolchain

namespace language {

class Type {
};

class SubstitutionMap {
private:
  void *storage = nullptr;

public:
  toolchain::ArrayRef<Type> getReplacementTypes() const;
};

} // namespace language

class BridgedArrayRef {
public:
  const void * Data;
  size_t Length;

  BridgedArrayRef() : Data(nullptr), Length(0) {}

#ifdef USED_IN_CPP_SOURCE
  template <typename T>
  BridgedArrayRef(toolchain::ArrayRef<T> arr)
      : Data(arr.Data), Length(arr.Length) {}

  template <typename T>
  toolchain::ArrayRef<T> unbridged() const {
    return {static_cast<const T *>(Data), Length};
  }
#endif
};

struct BridgedSubstitutionMap {
  uint64_t storage[1];

#ifdef USED_IN_CPP_SOURCE
  BridgedSubstitutionMap(language::SubstitutionMap map) {
    *reinterpret_cast<language::SubstitutionMap *>(&storage) = map;
  }
  language::SubstitutionMap unbridged() const {
    return *reinterpret_cast<const language::SubstitutionMap *>(&storage);
  }
#endif

  BridgedSubstitutionMap() {}
};

struct BridgedTypeArray {
  BridgedArrayRef typeArray;

#ifdef AFTER_FIX
   BridgedTypeArray() : typeArray() {}
#endif

#ifdef USED_IN_CPP_SOURCE
  BridgedTypeArray(toolchain::ArrayRef<language::Type> types) : typeArray(types) {}

  toolchain::ArrayRef<language::Type> unbridged() const {
    return typeArray.unbridged<language::Type>();
  }
#endif

  static BridgedTypeArray fromReplacementTypes(BridgedSubstitutionMap substMap);
};

#endif // TEST_INTEROP_CXX_CLASS_METHOD_SRET_WIN_ARM64_H
