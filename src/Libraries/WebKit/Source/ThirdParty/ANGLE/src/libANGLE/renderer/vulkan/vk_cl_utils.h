/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// vk_cl_utils:
//    Helper functions for the Vulkan Renderer in translation of vk state from/to cl state.
//

#ifndef LIBANGLE_RENDERER_VULKAN_CL_VK_UTILS_H_
#define LIBANGLE_RENDERER_VULKAN_CL_VK_UTILS_H_

#include "common/PackedCLEnums_autogen.h"

#include "libANGLE/CLBitField.h"
#include "libANGLE/cl_types.h"

#include "vulkan/vulkan_core.h"

namespace rx
{
namespace cl_vk
{
VkExtent3D GetExtent(const cl::Extents &extent);
VkOffset3D GetOffset(const cl::Offset &offset);
VkImageType GetImageType(cl::MemObjectType memObjectType);
VkImageViewType GetImageViewType(cl::MemObjectType memObjectType);
VkMemoryPropertyFlags GetMemoryPropertyFlags(cl::MemFlags memFlags);
VkBufferUsageFlags GetBufferUsageFlags(cl::MemFlags memFlags);

}  // namespace cl_vk
}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_CL_VK_UTILS_H_
