/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef SAMPLE_UTIL_TGA_UTILS_HPP
#define SAMPLE_UTIL_TGA_UTILS_HPP

#include <array>
#include <string>
#include <vector>

#include "util/gles_loader_autogen.h"

typedef std::array<unsigned char, 4> Byte4;

struct TGAImage
{
    size_t width;
    size_t height;
    std::vector<Byte4> data;

    TGAImage();
};

bool LoadTGAImageFromFile(const std::string &path, TGAImage *image);
GLuint LoadTextureFromTGAImage(const TGAImage &image);

#endif  // SAMPLE_UTIL_TGA_UTILS_HPP
