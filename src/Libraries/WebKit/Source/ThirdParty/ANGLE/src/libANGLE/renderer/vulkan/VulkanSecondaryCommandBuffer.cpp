/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// VulkanSecondaryCommandBuffer:
//    Implementation of VulkanSecondaryCommandBuffer.
//

#include "libANGLE/renderer/vulkan/VulkanSecondaryCommandBuffer.h"

#include "common/debug.h"
#include "libANGLE/renderer/vulkan/ContextVk.h"
#include "libANGLE/renderer/vulkan/vk_utils.h"

namespace rx
{
namespace vk
{
angle::Result VulkanSecondaryCommandBuffer::InitializeCommandPool(ErrorContext *context,
                                                                  SecondaryCommandPool *pool,
                                                                  uint32_t queueFamilyIndex,
                                                                  ProtectionType protectionType)
{
    ANGLE_TRY(pool->init(context, queueFamilyIndex, protectionType));
    return angle::Result::Continue;
}

angle::Result VulkanSecondaryCommandBuffer::InitializeRenderPassInheritanceInfo(
    ContextVk *contextVk,
    const Framebuffer &framebuffer,
    const RenderPassDesc &renderPassDesc,
    VkCommandBufferInheritanceInfo *inheritanceInfoOut,
    VkCommandBufferInheritanceRenderingInfo *renderingInfoOut,
    gl::DrawBuffersArray<VkFormat> *colorFormatStorageOut)
{
    *inheritanceInfoOut       = {};
    inheritanceInfoOut->sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;

    if (contextVk->getFeatures().preferDynamicRendering.enabled)
    {
        renderPassDesc.populateRenderingInheritanceInfo(contextVk->getRenderer(), renderingInfoOut,
                                                        colorFormatStorageOut);
        AddToPNextChain(inheritanceInfoOut, renderingInfoOut);
    }
    else
    {
        const RenderPass *compatibleRenderPass = nullptr;
        ANGLE_TRY(contextVk->getCompatibleRenderPass(renderPassDesc, &compatibleRenderPass));
        inheritanceInfoOut->renderPass  = compatibleRenderPass->getHandle();
        inheritanceInfoOut->subpass     = 0;
        inheritanceInfoOut->framebuffer = framebuffer.getHandle();
    }

    return angle::Result::Continue;
}

angle::Result VulkanSecondaryCommandBuffer::initialize(ErrorContext *context,
                                                       SecondaryCommandPool *pool,
                                                       bool isRenderPassCommandBuffer,
                                                       SecondaryCommandMemoryAllocator *allocator)
{
    mCommandPool = pool;
    mCommandTracker.reset();
    mAnyCommand = false;

    ANGLE_TRY(pool->allocate(context, this));

    // Outside-RP command buffers are begun automatically here.  RP command buffers are begun when
    // the render pass itself starts, as they require inheritance info.
    if (!isRenderPassCommandBuffer)
    {
        VkCommandBufferInheritanceInfo inheritanceInfo = {};
        inheritanceInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        ANGLE_TRY(begin(context, inheritanceInfo));
    }

    return angle::Result::Continue;
}

void VulkanSecondaryCommandBuffer::destroy()
{
    if (valid())
    {
        ASSERT(mCommandPool != nullptr);
        mCommandPool->collect(this);
    }
}

angle::Result VulkanSecondaryCommandBuffer::begin(
    ErrorContext *context,
    const VkCommandBufferInheritanceInfo &inheritanceInfo)
{
    ASSERT(!mAnyCommand);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    beginInfo.pInheritanceInfo         = &inheritanceInfo;
    // For render pass command buffers, specify that the command buffer is entirely inside a render
    // pass.  This is determined by either the render pass object existing, or the
    // VkCommandBufferInheritanceRenderingInfo specified in the pNext chain.
    if (inheritanceInfo.renderPass != VK_NULL_HANDLE || inheritanceInfo.pNext != nullptr)
    {
        beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    }

    ANGLE_VK_TRY(context, CommandBuffer::begin(beginInfo));
    return angle::Result::Continue;
}

angle::Result VulkanSecondaryCommandBuffer::end(ErrorContext *context)
{
    ANGLE_VK_TRY(context, CommandBuffer::end());
    return angle::Result::Continue;
}

VkResult VulkanSecondaryCommandBuffer::reset()
{
    mCommandTracker.reset();
    mAnyCommand = false;
    return CommandBuffer::reset();
}
}  // namespace vk
}  // namespace rx
