/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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

#ifndef ANGLE_TESTS_ANGLE_TEST_PLATFORM_H_
#define ANGLE_TESTS_ANGLE_TEST_PLATFORM_H_

#include <string>

#include "util/util_gl.h"

// Driver vendors
bool IsAdreno();

// Renderer back-ends
bool IsD3D11();
// Is a D3D9-class renderer.
bool IsD3D9();
bool IsDesktopOpenGL();
bool IsOpenGLES();
bool IsOpenGL();
bool IsNULL();
bool IsVulkan();
bool IsMetal();
bool IsD3D();

// Debug/Release
bool IsDebug();
bool IsRelease();

bool EnsureGLExtensionEnabled(const std::string &extName);
bool IsEGLClientExtensionEnabled(const std::string &extName);
bool IsEGLDeviceExtensionEnabled(EGLDeviceEXT device, const std::string &extName);
bool IsEGLDisplayExtensionEnabled(EGLDisplay display, const std::string &extName);
bool IsGLExtensionEnabled(const std::string &extName);
bool IsGLExtensionRequestable(const std::string &extName);

#endif  // ANGLE_TESTS_ANGLE_TEST_PLATFORM_H_
