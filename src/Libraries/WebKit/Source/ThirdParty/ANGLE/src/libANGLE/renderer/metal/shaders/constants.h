/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
// Copyright 2020 The ANGLE Project. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// constants.h: Declare some constant values to be used by metal defaultshaders.

#ifndef LIBANGLE_RENDERER_METAL_SHADERS_ENUM_H_
#define LIBANGLE_RENDERER_METAL_SHADERS_ENUM_H_

namespace rx
{
namespace mtl_shader
{

enum
{
    kTextureType2D            = 0,
    kTextureType2DMultisample = 1,
    kTextureType2DArray       = 2,
    kTextureTypeCube          = 3,
    kTextureType3D            = 4,
    kTextureTypeCount         = 5,
};

// Metal doesn't support constexpr to be used as array size, so we need to use macro here
#define kGenerateMipThreadGroupSizePerDim 8

}  // namespace mtl_shader
}  // namespace rx

#endif
