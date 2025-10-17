/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// driver_utils_d3d.cpp: Information specific to the D3D driver

#include "libANGLE/renderer/d3d/driver_utils_d3d.h"

namespace rx
{

std::string GetDriverVersionString(LARGE_INTEGER driverVersion)
{
    std::stringstream versionString;
    uint64_t intVersion        = driverVersion.QuadPart;
    constexpr uint64_t kMask16 = std::numeric_limits<uint16_t>::max();
    versionString << ((intVersion >> 48) & kMask16) << ".";
    versionString << ((intVersion >> 32) & kMask16) << ".";
    versionString << ((intVersion >> 16) & kMask16) << ".";
    versionString << (intVersion & kMask16);
    return versionString.str();
}

}  // namespace rx
