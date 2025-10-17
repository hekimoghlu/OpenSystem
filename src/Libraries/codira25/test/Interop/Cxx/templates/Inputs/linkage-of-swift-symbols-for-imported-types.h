/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_LINKAGE_OF_LANGUAGE_SYMBOLS_FOR_IMPORTED_TYPES_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_LINKAGE_OF_LANGUAGE_SYMBOLS_FOR_IMPORTED_TYPES_H

template<class T>
struct MagicWrapper {
  T t;
  int callGetInt() const {
    return t.getInt() + 5;
  }
};

struct MagicNumber {
  // Codira runtime defines many value witness tables for types with some common layouts.
  // This struct's uncommon size forces the compiler to define a new value witness table instead of reusing one from the runtime.
  char forceVWTableCreation[57];
  int getInt() const { return 12; }
};

typedef MagicWrapper<MagicNumber> WrappedMagicNumber;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_LINKAGE_OF_LANGUAGE_SYMBOLS_FOR_IMPORTED_TYPES_H
