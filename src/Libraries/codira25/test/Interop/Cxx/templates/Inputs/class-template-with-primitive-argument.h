/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_WITH_PRIMITIVE_ARGUMENT_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_WITH_PRIMITIVE_ARGUMENT_H

#include <cstddef>

template<class T>
struct MagicWrapper {
  T t;
  int getValuePlusArg(int arg) const { return t + arg; }
};

template<class M>
struct DoubleWrapper {
  M m;
  int getValuePlusArg(int arg) const { return m.getValuePlusArg(arg); }
};

typedef MagicWrapper<int> WrappedMagicInt;
typedef MagicWrapper<const int> WrappedMagicIntConst;
typedef MagicWrapper<const long> WrappedMagicLongConst;
typedef MagicWrapper<int*> WrappedMagicIntPtr;
typedef MagicWrapper<const int*> WrappedMagicIntConstPtr;
typedef MagicWrapper<int**> WrappedMagicIntPtrPtr;
typedef MagicWrapper<int[]> WrappedMagicIntArr;
typedef MagicWrapper<long[]> WrappedMagicLongArr;
typedef MagicWrapper<int[123]> WrappedMagicIntFixedSizeArr1;
typedef MagicWrapper<int[124]> WrappedMagicIntFixedSizeArr2;
typedef MagicWrapper<std::nullptr_t> WrappedMagicNullPtr;

typedef DoubleWrapper<MagicWrapper<int>> DoubleWrappedInt;
typedef DoubleWrapper<MagicWrapper<const int>> DoubleWrappedIntConst;
typedef DoubleWrapper<MagicWrapper<const long>> DoubleWrappedLongConst;
typedef DoubleWrapper<MagicWrapper<int*>> DoubleWrappedIntPtr;
typedef DoubleWrapper<MagicWrapper<const int*>> DoubleWrappedIntConstPtr;
typedef DoubleWrapper<MagicWrapper<int[]>> DoubleWrappedMagicIntArr;
typedef DoubleWrapper<MagicWrapper<long[]>> DoubleWrappedMagicLongArr;
typedef DoubleWrapper<MagicWrapper<int[42]>> DoubleWrappedMagicIntFixedSizeArr1;
typedef DoubleWrapper<MagicWrapper<int[43]>> DoubleWrappedMagicIntFixedSizeArr2;
typedef DoubleWrapper<MagicWrapper<std::nullptr_t>> DoubleWrappedMagicNullPtr;

typedef DoubleWrapper<const MagicWrapper<int>> DoubleConstWrappedInt;
typedef DoubleWrapper<const MagicWrapper<const int>> DoubleConstWrappedIntConst;
typedef DoubleWrapper<const MagicWrapper<const long>> DoubleConstWrappedLongConst;
typedef DoubleWrapper<const MagicWrapper<int*>> DoubleConstWrappedIntPtr;
typedef DoubleWrapper<const MagicWrapper<const int*>> DoubleConstWrappedIntConstPtr;
typedef DoubleWrapper<const MagicWrapper<int[]>> DoubleConstWrappedMagicIntArr;
typedef DoubleWrapper<const MagicWrapper<long[]>> DoubleConstWrappedMagicLongArr;
typedef DoubleWrapper<const MagicWrapper<int[42]>> DoubleConstWrappedMagicIntFixedSizeArr1;
typedef DoubleWrapper<const MagicWrapper<int[43]>> DoubleConstWrappedMagicIntFixedSizeArr2;
typedef DoubleWrapper<const MagicWrapper<std::nullptr_t>> DoubleConstWrappedMagicNullPtr;

typedef MagicWrapper<volatile int> WrappedVolatileInt;
typedef MagicWrapper<const volatile int> WrappedConstVolatileInt;
typedef MagicWrapper<volatile const int> WrappedVolatileConstInt;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_WITH_PRIMITIVE_ARGUMENT_H
