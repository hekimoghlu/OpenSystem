/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

// Color.h : Defines the Color type used throughout the ANGLE libraries

#ifndef COMMON_COLOR_H_
#define COMMON_COLOR_H_

#include <cstdint>
#include <cstring>

#include "common/debug.h"

namespace angle
{

template <typename T>
struct Color
{
    Color();
    constexpr Color(T r, T g, T b, T a);

    const T *data() const { return &red; }
    T *ptr() { return &red; }

    static Color fromData(const T *data) { return Color(data[0], data[1], data[2], data[3]); }
    void writeData(T *data) const
    {
        data[0] = red;
        data[1] = green;
        data[2] = blue;
        data[3] = alpha;
    }

    T red;
    T green;
    T blue;
    T alpha;
};

template <typename T>
bool operator==(const Color<T> &a, const Color<T> &b);

template <typename T>
bool operator!=(const Color<T> &a, const Color<T> &b);

typedef Color<float> ColorF;
typedef Color<int> ColorI;
typedef Color<unsigned int> ColorUI;

ANGLE_ENABLE_STRUCT_PADDING_WARNINGS
struct ColorGeneric
{
    inline ColorGeneric();
    inline ColorGeneric(const ColorF &color);
    inline ColorGeneric(const ColorI &color);
    inline ColorGeneric(const ColorUI &color);

    enum class Type : uint32_t
    {
        Float = 0,
        Int   = 1,
        UInt  = 2
    };

    union
    {
        ColorF colorF;
        ColorI colorI;
        ColorUI colorUI;
    };

    Type type;
};
ANGLE_DISABLE_STRUCT_PADDING_WARNINGS

inline bool operator==(const ColorGeneric &a, const ColorGeneric &b);

inline bool operator!=(const ColorGeneric &a, const ColorGeneric &b);

struct DepthStencil
{
    DepthStencil() : depth(0), stencil(0) {}

    // Double is needed to represent the 32-bit integer range of GL_DEPTH_COMPONENT32.
    double depth;
    uint32_t stencil;
};
}  // namespace angle

// TODO: Move this fully into the angle namespace
namespace gl
{

template <typename T>
using Color        = angle::Color<T>;
using ColorF       = angle::ColorF;
using ColorI       = angle::ColorI;
using ColorUI      = angle::ColorUI;
using ColorGeneric = angle::ColorGeneric;

}  // namespace gl

#include "Color.inc"

#endif  // COMMON_COLOR_H_
