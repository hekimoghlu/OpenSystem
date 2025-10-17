/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// This file includes APIs to detect whether certain Apple renderer is available for testing.
//

#include "test_utils/angle_test_instantiate_apple.h"

#include "common/apple_platform_utils.h"
#include "test_utils/angle_test_instantiate.h"

namespace angle
{

bool IsMetalTextureSwizzleAvailable()
{
#if ANGLE_PLATFORM_IOS_FAMILY_SIMULATOR
    return false;
#else
    // All NVIDIA and older Intel don't support swizzle because they are GPU family 1.
    // We don't have a way to detect Metal family here, so skip all Intel for now.
    return !IsIntel() && !IsNVIDIA();
#endif
}

}  // namespace angle
