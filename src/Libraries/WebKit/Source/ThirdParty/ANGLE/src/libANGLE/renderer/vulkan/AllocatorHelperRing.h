/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
// AllocatorHelperRing:
//    Manages the ring buffer allocators used in the command buffers.
//

#ifndef LIBANGLE_RENDERER_VULKAN_ALLOCATORHELPERRING_H_
#define LIBANGLE_RENDERER_VULKAN_ALLOCATORHELPERRING_H_

#include "common/RingBufferAllocator.h"
#include "common/vulkan/vk_headers.h"
#include "libANGLE/renderer/vulkan/vk_command_buffer_utils.h"
#include "libANGLE/renderer/vulkan/vk_wrapper.h"

namespace rx
{
namespace vk
{
namespace priv
{
class SecondaryCommandBuffer;
}  // namespace priv

using SharedCommandMemoryAllocator = angle::SharedRingBufferAllocator;
using RingBufferAllocator          = angle::RingBufferAllocator;

// Used in CommandBufferHelperCommon
class SharedCommandBlockAllocator
{
  public:
    SharedCommandBlockAllocator() : mAllocator(nullptr), mAllocSharedCP(nullptr) {}
    void resetAllocator();
    bool hasAllocatorLinks() const { return mAllocator || mAllocSharedCP; }

    void init() {}

    void attachAllocator(SharedCommandMemoryAllocator *allocator);
    SharedCommandMemoryAllocator *detachAllocator(bool isCommandBufferEmpty);

    SharedCommandMemoryAllocator *getAllocator() { return mAllocator; }

  private:
    // Using a ring buffer allocator for less memory overhead (observed with the async queue
    // ex-feature)
    SharedCommandMemoryAllocator *mAllocator;
    angle::SharedRingBufferAllocatorCheckPoint *mAllocSharedCP;
    angle::RingBufferAllocatorCheckPoint mAllocReleaseCP;
};

// Used in SecondaryCommandBuffer
class SharedCommandBlockPool final : angle::RingBufferAllocateListener
{
  public:
    SharedCommandBlockPool()
        : mLastCommandBlock(nullptr), mFinishedCommandSize(0), mCommandBuffer(nullptr)
    {}

    static constexpr size_t kCommandHeaderSize = 4;
    using CommandHeaderIDType                  = uint16_t;
    // Make sure the size of command header ID type is less than total command header size.
    static_assert(sizeof(CommandHeaderIDType) < kCommandHeaderSize, "Check size of CommandHeader");

    void setCommandBuffer(priv::SecondaryCommandBuffer *commandBuffer)
    {
        mCommandBuffer = commandBuffer;
    }
    void resetCommandBuffer() { mCommandBuffer = nullptr; }

    void reset(CommandBufferCommandTracker *commandBufferTracker)
    {
        mLastCommandBlock    = nullptr;
        mFinishedCommandSize = 0;
        if (mAllocator.valid())
        {
            mAllocator.release(mAllocator.getReleaseCheckPoint());
            pushNewCommandBlock(mAllocator.allocate(0));
        }
    }

    angle::Result initialize(SharedCommandMemoryAllocator *allocator)
    {
        return angle::Result::Continue;
    }

    // Always valid (even if allocator is detached).
    bool valid() const { return true; }
    bool empty() const { return getCommandSize() == 0; }

    void getMemoryUsageStats(size_t *usedMemoryOut, size_t *allocatedMemoryOut) const;

    void terminateLastCommandBlock()
    {
        if (mLastCommandBlock)
        {
            ASSERT(mAllocator.valid());
            ASSERT(mAllocator.getPointer() >= mLastCommandBlock);
            ASSERT(mAllocator.getFragmentSize() >= kCommandHeaderSize);
            reinterpret_cast<CommandHeaderIDType &>(*(mAllocator.getPointer())) = 0;
        }
    }

    // Initialize the SecondaryCommandBuffer by setting the allocator it will use
    void attachAllocator(SharedCommandMemoryAllocator *source);
    void detachAllocator(SharedCommandMemoryAllocator *destination);

    void onNewVariableSizedCommand(const size_t requiredSize,
                                   const size_t allocationSize,
                                   uint8_t **headerOut)
    {
        *headerOut = allocateCommand(allocationSize);
    }
    void onNewCommand(const size_t requiredSize, const size_t allocationSize, uint8_t **headerOut)
    {
        *headerOut = allocateCommand(allocationSize);
    }

  private:
    uint8_t *allocateCommand(size_t allocationSize)
    {
        ASSERT(mLastCommandBlock);
        return mAllocator.allocate(static_cast<uint32_t>(allocationSize));
    }

    // The following is used to give the size of the command buffer in bytes
    uint32_t getCommandSize() const
    {
        uint32_t result = mFinishedCommandSize;
        if (mLastCommandBlock)
        {
            ASSERT(mAllocator.valid());
            ASSERT(mAllocator.getPointer() >= mLastCommandBlock);
            result += static_cast<uint32_t>(mAllocator.getPointer() - mLastCommandBlock);
        }
        return result;
    }

    void pushNewCommandBlock(uint8_t *block);
    void finishLastCommandBlock();

    // Functions derived from RingBufferAllocateListener
    virtual void onRingBufferNewFragment() override;
    virtual void onRingBufferFragmentEnd() override;

    // Using a ring buffer allocator for less memory overhead (observed with the async queue
    // ex-feature)
    RingBufferAllocator mAllocator;
    uint8_t *mLastCommandBlock;
    uint32_t mFinishedCommandSize;

    // Points to the parent command buffer.
    priv::SecondaryCommandBuffer *mCommandBuffer;
};

}  // namespace vk
}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_ALLOCATORHELPERRING_H_
