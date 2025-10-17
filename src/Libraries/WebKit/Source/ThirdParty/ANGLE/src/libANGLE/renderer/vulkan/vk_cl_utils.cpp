/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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

#include "libANGLE/renderer/vulkan/vk_cl_utils.h"
#include "vulkan/vulkan_core.h"

namespace rx
{
namespace cl_vk
{

VkExtent3D GetExtent(const cl::Extents &extent)
{
    VkExtent3D vkExtent{};

    vkExtent.width  = static_cast<uint32_t>(extent.width);
    vkExtent.height = static_cast<uint32_t>(extent.height);
    vkExtent.depth  = static_cast<uint32_t>(extent.depth);

    return vkExtent;
}

VkOffset3D GetOffset(const cl::Offset &offset)
{
    VkOffset3D vkOffset{};

    vkOffset.x = static_cast<uint32_t>(offset.x);
    vkOffset.y = static_cast<uint32_t>(offset.y);
    vkOffset.z = static_cast<uint32_t>(offset.z);

    return vkOffset;
}

VkImageType GetImageType(cl::MemObjectType memObjectType)
{
    switch (memObjectType)
    {
        case cl::MemObjectType::Image1D:
        case cl::MemObjectType::Image1D_Array:
        case cl::MemObjectType::Image1D_Buffer:
            return VK_IMAGE_TYPE_1D;
        case cl::MemObjectType::Image2D:
        case cl::MemObjectType::Image2D_Array:
            return VK_IMAGE_TYPE_2D;
        case cl::MemObjectType::Image3D:
            return VK_IMAGE_TYPE_3D;
        default:
            // We will need to implement all the texture types for ES3+.
            UNIMPLEMENTED();
            return VK_IMAGE_TYPE_MAX_ENUM;
    }
}

VkImageViewType GetImageViewType(cl::MemObjectType memObjectType)
{
    switch (memObjectType)
    {
        case cl::MemObjectType::Image1D:
            return VK_IMAGE_VIEW_TYPE_1D;
        case cl::MemObjectType::Image1D_Array:
            return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        case cl::MemObjectType::Image2D:
            return VK_IMAGE_VIEW_TYPE_2D;
        case cl::MemObjectType::Image2D_Array:
            return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        case cl::MemObjectType::Image3D:
            return VK_IMAGE_VIEW_TYPE_3D;
        case cl::MemObjectType::Image1D_Buffer:
            // Image1D_Buffer has an associated buffer view and not an image view, returning max
            // enum here.
            return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
        default:
            UNIMPLEMENTED();
            return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
    }
}

VkMemoryPropertyFlags GetMemoryPropertyFlags(cl::MemFlags memFlags)
{
    // TODO: http://anglebug.com/42267018
    VkMemoryPropertyFlags propFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

    if (memFlags.intersects(CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR))
    {
        propFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    }

    return propFlags;
}

VkBufferUsageFlags GetBufferUsageFlags(cl::MemFlags memFlags)
{
    // The buffer usage flags don't particularly affect the buffer in any known drivers, use all the
    // bits that ANGLE needs.
    return VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
           VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
}

}  // namespace cl_vk
}  // namespace rx
