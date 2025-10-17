/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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

// copyimage.h: Defines image copying functions

#ifndef IMAGEUTIL_COPYIMAGE_H_
#define IMAGEUTIL_COPYIMAGE_H_

#include "common/Color.h"

#include "image_util/imageformats.h"

#include <stdint.h>

namespace angle
{

template <typename sourceType, typename colorDataType>
void ReadColor(const uint8_t *source, uint8_t *dest);

template <typename destType, typename colorDataType>
void WriteColor(const uint8_t *source, uint8_t *dest);

template <typename SourceType>
void ReadDepthStencil(const uint8_t *source, uint8_t *dest);

template <typename DestType>
void WriteDepthStencil(const uint8_t *source, uint8_t *dest);

void CopyBGRA8ToRGBA8(const uint8_t *source,
                      int srcXAxisPitch,
                      int srcYAxisPitch,
                      uint8_t *dest,
                      int destXAxisPitch,
                      int destYAxisPitch,
                      int destWidth,
                      int destHeight);

}  // namespace angle

#include "copyimage.inc"

#endif  // IMAGEUTIL_COPYIMAGE_H_
