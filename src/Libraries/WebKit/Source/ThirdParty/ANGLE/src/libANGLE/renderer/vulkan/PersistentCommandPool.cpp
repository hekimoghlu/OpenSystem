/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
// PersistentCommandPool.cpp:
//    Implements the class methods for PersistentCommandPool
//

#include "libANGLE/renderer/vulkan/PersistentCommandPool.h"

namespace rx
{

namespace vk
{

PersistentCommandPool::PersistentCommandPool() {}

PersistentCommandPool::~PersistentCommandPool()
{
    ASSERT(!mCommandPool.valid() && mFreeBuffers.empty());
}

angle::Result PersistentCommandPool::init(ErrorContext *context,
                                          ProtectionType protectionType,
                                          uint32_t queueFamilyIndex)
{
    ASSERT(!mCommandPool.valid());

    // Initialize the command pool now that we know the queue family index.
    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // TODO (https://issuetracker.google.com/issues/166793850) We currently reset individual
    //  command buffers from this pool. Alternatively we could reset the entire command pool.
    commandPoolInfo.flags =
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    ASSERT(protectionType == ProtectionType::Unprotected ||
           protectionType == ProtectionType::Protected);
    if (protectionType == ProtectionType::Protected)
    {
        commandPoolInfo.flags |= VK_COMMAND_POOL_CREATE_PROTECTED_BIT;
    }
    commandPoolInfo.queueFamilyIndex = queueFamilyIndex;

    ANGLE_VK_TRY(context, mCommandPool.init(context->getDevice(), commandPoolInfo));

    for (uint32_t i = 0; i < kInitBufferNum; i++)
    {
        ANGLE_TRY(allocateCommandBuffer(context));
    }

    return angle::Result::Continue;
}

void PersistentCommandPool::destroy(VkDevice device)
{
    if (!valid())
        return;

    ASSERT(mCommandPool.valid());

    for (PrimaryCommandBuffer &cmdBuf : mFreeBuffers)
    {
        cmdBuf.destroy(device, mCommandPool);
    }
    mFreeBuffers.clear();

    mCommandPool.destroy(device);
}

angle::Result PersistentCommandPool::allocate(ErrorContext *context,
                                              PrimaryCommandBuffer *commandBufferOut)
{
    if (mFreeBuffers.empty())
    {
        ANGLE_TRY(allocateCommandBuffer(context));
        ASSERT(!mFreeBuffers.empty());
    }

    *commandBufferOut = std::move(mFreeBuffers.back());
    mFreeBuffers.pop_back();

    return angle::Result::Continue;
}

angle::Result PersistentCommandPool::collect(ErrorContext *context, PrimaryCommandBuffer &&buffer)
{
    // VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT NOT set, The CommandBuffer
    // can still hold the memory resource
    ANGLE_VK_TRY(context, buffer.reset());

    mFreeBuffers.emplace_back(std::move(buffer));
    return angle::Result::Continue;
}

angle::Result PersistentCommandPool::allocateCommandBuffer(ErrorContext *context)
{
    PrimaryCommandBuffer commandBuffer;
    {
        // Only used for primary CommandBuffer allocation
        VkCommandBufferAllocateInfo commandBufferInfo = {};
        commandBufferInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool        = mCommandPool.getHandle();
        commandBufferInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1;

        ANGLE_VK_TRY(context, commandBuffer.init(context->getDevice(), commandBufferInfo));
    }

    mFreeBuffers.emplace_back(std::move(commandBuffer));

    return angle::Result::Continue;
}

}  // namespace vk

}  // namespace rx
