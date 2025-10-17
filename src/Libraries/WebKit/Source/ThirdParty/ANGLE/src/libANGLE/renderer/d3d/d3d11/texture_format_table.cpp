/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
// Helper routines for the D3D11 texture format table.

#include "libANGLE/renderer/d3d/d3d11/texture_format_table.h"

#include "libANGLE/renderer/load_functions_table.h"

namespace rx
{

namespace d3d11
{

const Format &Format::getSwizzleFormat(const Renderer11DeviceCaps &deviceCaps) const
{
    return (swizzleFormat == internalFormat ? *this : Format::Get(swizzleFormat, deviceCaps));
}

LoadFunctionMap Format::getLoadFunctions() const
{
    return GetLoadFunctionsMap(internalFormat, formatID);
}

const angle::Format &Format::format() const
{
    return angle::Format::Get(formatID);
}

}  // namespace d3d11

}  // namespace rx
