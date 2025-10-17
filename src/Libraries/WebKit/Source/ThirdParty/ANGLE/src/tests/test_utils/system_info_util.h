/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// system_info_util.h:
//   Implementation of common test utilities for operating with SystemInfo.
//

#ifndef ANGLE_TESTS_SYSTEM_INFO_UTIL_H_
#define ANGLE_TESTS_SYSTEM_INFO_UTIL_H_

#include <stddef.h>

namespace angle
{
struct SystemInfo;
}  // namespace angle

// Returns the index of the low power GPU in SystemInfo.
size_t FindLowPowerGPU(const angle::SystemInfo &);

// Returns the index of the high power GPU in SystemInfo.
size_t FindHighPowerGPU(const angle::SystemInfo &);

// Returns the index of the GPU in SystemInfo based on the OpenGL renderer string.
size_t FindActiveOpenGLGPU(const angle::SystemInfo &);

#endif  // ANGLE_TESTS_SYSTEM_INFO_UTIL_H_
