/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// copyvertex.h: Defines vertex buffer copying and conversion functions

#ifndef LIBANGLE_RENDERER_COPYVERTEX_H_
#define LIBANGLE_RENDERER_COPYVERTEX_H_

#include "common/mathutil.h"

namespace rx
{

using VertexCopyFunction = void (*)(const uint8_t *input,
                                    size_t stride,
                                    size_t count,
                                    uint8_t *output);

// 'alphaDefaultValueBits' gives the default value for the alpha channel (4th component)
template <typename T,
          size_t inputComponentCount,
          size_t outputComponentCount,
          uint32_t alphaDefaultValueBits>
void CopyNativeVertexData(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

template <size_t inputComponentCount, size_t outputComponentCount>
void Copy8SintTo16SintVertexData(const uint8_t *input,
                                 size_t stride,
                                 size_t count,
                                 uint8_t *output);

template <size_t componentCount>
void Copy8SnormTo16SnormVertexData(const uint8_t *input,
                                   size_t stride,
                                   size_t count,
                                   uint8_t *output);

template <size_t inputComponentCount, size_t outputComponentCount>
void Copy32FixedTo32FVertexData(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

template <typename T,
          size_t inputComponentCount,
          size_t outputComponentCount,
          bool normalized,
          bool toHalf>
void CopyToFloatVertexData(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

template <size_t inputComponentCount, size_t outputComponentCount>
void Copy32FTo16FVertexData(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

void CopyXYZ32FToXYZ9E5(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

void CopyXYZ32FToX11Y11B10F(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

template <bool isSigned, bool normalized, bool toFloat, bool toHalf>
void CopyXYZ10W2ToXYZWFloatVertexData(const uint8_t *input,
                                      size_t stride,
                                      size_t count,
                                      uint8_t *output);

template <bool isSigned, bool normalized, bool toHalf>
void CopyXYZ10ToXYZWFloatVertexData(const uint8_t *input,
                                    size_t stride,
                                    size_t count,
                                    uint8_t *output);

template <bool isSigned, bool normalized, bool toHalf>
void CopyW2XYZ10ToXYZWFloatVertexData(const uint8_t *input,
                                      size_t stride,
                                      size_t count,
                                      uint8_t *output);

}  // namespace rx

#include "copyvertex.inc.h"

#endif  // LIBANGLE_RENDERER_COPYVERTEX_H_
