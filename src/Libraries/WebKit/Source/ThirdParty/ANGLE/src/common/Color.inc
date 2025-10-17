/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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

//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Color.inc : Inline definitions of some functions from Color.h

namespace angle
{

template <typename T>
Color<T>::Color() : Color(0, 0, 0, 0)
{
}

template <typename T>
constexpr Color<T>::Color(T r, T g, T b, T a) : red(r), green(g), blue(b), alpha(a)
{
}

template <typename T>
bool operator==(const Color<T> &a, const Color<T> &b)
{
    return a.red == b.red &&
           a.green == b.green &&
           a.blue == b.blue &&
           a.alpha == b.alpha;
}

template <typename T>
bool operator!=(const Color<T> &a, const Color<T> &b)
{
    return !(a == b);
}


ColorGeneric::ColorGeneric() : colorF(), type(Type::Float) {}

ColorGeneric::ColorGeneric(const ColorF &color) : colorF(color), type(Type::Float) {}

ColorGeneric::ColorGeneric(const ColorI &color) : colorI(color), type(Type::Int) {}

ColorGeneric::ColorGeneric(const ColorUI &color) : colorUI(color), type(Type::UInt) {}

bool operator==(const ColorGeneric &a, const ColorGeneric &b)
{
    // Switching on the type and comparing individual
    // channels yield much less efficient code.
    return memcmp(&a, &b, sizeof(ColorGeneric)) == 0;
}

bool operator!=(const ColorGeneric &a, const ColorGeneric &b)
{
    return !(a == b);
}

}  // namespace angle
