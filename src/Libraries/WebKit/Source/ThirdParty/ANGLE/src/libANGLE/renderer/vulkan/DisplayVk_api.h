/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
// DisplayVk_api.h:
//    Defines the Vulkan Display APIs to be used by the code outside the back-end.
//

#ifndef LIBANGLE_RENDERER_VULKAN_DISPLAYVK_API_H_
#define LIBANGLE_RENDERER_VULKAN_DISPLAYVK_API_H_

#include "libANGLE/renderer/DisplayImpl.h"

namespace rx
{

bool IsVulkanNullDisplayAvailable();
DisplayImpl *CreateVulkanNullDisplay(const egl::DisplayState &state);

#if defined(ANGLE_PLATFORM_WINDOWS)
bool IsVulkanWin32DisplayAvailable();
DisplayImpl *CreateVulkanWin32Display(const egl::DisplayState &state);
#endif  // defined(ANGLE_PLATFORM_WINDOWS)

#if defined(ANGLE_PLATFORM_LINUX)
bool IsVulkanWaylandDisplayAvailable();
DisplayImpl *CreateVulkanWaylandDisplay(const egl::DisplayState &state);

bool IsVulkanGbmDisplayAvailable();
DisplayImpl *CreateVulkanGbmDisplay(const egl::DisplayState &state);

bool IsVulkanXcbDisplayAvailable();
DisplayImpl *CreateVulkanXcbDisplay(const egl::DisplayState &state);

bool IsVulkanSimpleDisplayAvailable();
DisplayImpl *CreateVulkanSimpleDisplay(const egl::DisplayState &state);

bool IsVulkanHeadlessDisplayAvailable();
DisplayImpl *CreateVulkanHeadlessDisplay(const egl::DisplayState &state);

bool IsVulkanOffscreenDisplayAvailable();
DisplayImpl *CreateVulkanOffscreenDisplay(const egl::DisplayState &state);
#endif  // defined(ANGLE_PLATFORM_LINUX)

#if defined(ANGLE_PLATFORM_ANDROID)
bool IsVulkanAndroidDisplayAvailable();
DisplayImpl *CreateVulkanAndroidDisplay(const egl::DisplayState &state);
#endif  // defined(ANGLE_PLATFORM_ANDROID)

#if defined(ANGLE_PLATFORM_FUCHSIA)
bool IsVulkanFuchsiaDisplayAvailable();
DisplayImpl *CreateVulkanFuchsiaDisplay(const egl::DisplayState &state);
#endif  // defined(ANGLE_PLATFORM_FUCHSIA)

#if defined(ANGLE_PLATFORM_GGP)
bool IsVulkanGGPDisplayAvailable();
DisplayImpl *CreateVulkanGGPDisplay(const egl::DisplayState &state);
#endif  // defined(ANGLE_PLATFORM_GGP)

#if defined(ANGLE_PLATFORM_APPLE)
bool IsVulkanMacDisplayAvailable();
DisplayImpl *CreateVulkanMacDisplay(const egl::DisplayState &state);
#endif  // defined(ANGLE_PLATFORM_APPLE)
}  // namespace rx

#endif /* LIBANGLE_RENDERER_VULKAN_DISPLAYVK_API_H_ */
