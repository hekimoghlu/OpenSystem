/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// generatemip.h: Defines the GenerateMip function, templated on the format
// type of the image for which mip levels are being generated.

#ifndef IMAGEUTIL_GENERATEMIP_H_
#define IMAGEUTIL_GENERATEMIP_H_

#include <stddef.h>
#include <stdint.h>

namespace angle
{

template <typename T>
inline void GenerateMip(size_t sourceWidth,
                        size_t sourceHeight,
                        size_t sourceDepth,
                        const uint8_t *sourceData,
                        size_t sourceRowPitch,
                        size_t sourceDepthPitch,
                        uint8_t *destData,
                        size_t destRowPitch,
                        size_t destDepthPitch);

}  // namespace angle

#include "generatemip.inc"

#endif  // IMAGEUTIL_GENERATEMIP_H_
