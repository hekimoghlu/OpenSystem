/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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

// driver_utils_d3d.h: Information specific to the D3D driver

#ifndef LIBANGLE_RENDERER_D3D_DRIVER_UTILS_D3D_H_
#define LIBANGLE_RENDERER_D3D_DRIVER_UTILS_D3D_H_

#include "libANGLE/angletypes.h"

namespace rx
{

std::string GetDriverVersionString(LARGE_INTEGER driverVersion);

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_DRIVER_UTILS_D3D_H_
