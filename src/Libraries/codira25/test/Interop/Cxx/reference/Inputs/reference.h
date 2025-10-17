/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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

#ifndef TEST_INTEROP_CXX_REFERENCE_INPUTS_REFERENCE_H
#define TEST_INTEROP_CXX_REFERENCE_INPUTS_REFERENCE_H

int getStaticInt();
int &getStaticIntRef();
int &&getStaticIntRvalueRef();
const int &getConstStaticIntRef();
const int &&getConstStaticIntRvalueRef();

void setStaticInt(int);
void setStaticIntRef(int &);
void setStaticIntRvalueRef(int &&);
void setConstStaticIntRef(const int &);
void setConstStaticIntRvalueRef(const int &&);

auto getFuncRef() -> int (&)();
auto getFuncRvalueRef() -> int (&&)();

using ConstIntRefTypealias = const int &;

void setConstStaticIntRefTypealias(ConstIntRefTypealias ref);

using IntRefTypealias = int &;

void setStaticIntRefTypealias(IntRefTypealias ref);

template<class T>
struct ClassTemplate {};

template<class T>
const ClassTemplate<T> &refToDependent() { return ClassTemplate<T>(); }

// We cannot import "_Atomic" types. Make sure we fail gracefully instead of
// crashing when we have an "_Atomic" type or a reference to one.
void dontImportAtomicRef(_Atomic(int)&) { }

void takeConstRef(const int &);
inline bool takeConstRefBool(const bool &b) { return b; }
inline void takeRefBool(bool &b) { b = true; }

template<class T>
T &refToTemplate(T &t) { return t; }

template<class T>
const T &constRefToTemplate(const T &t) { return t; }

template<class T>
void refToDependentParam(ClassTemplate<T> &param) { }

#endif // TEST_INTEROP_CXX_REFERENCE_INPUTS_REFERENCE_H
