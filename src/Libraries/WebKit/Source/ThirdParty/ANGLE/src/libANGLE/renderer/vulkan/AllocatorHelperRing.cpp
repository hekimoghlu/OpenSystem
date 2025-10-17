/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
//    Implements the ring buffer allocator helpers used in the command buffers.
//

#include "libANGLE/renderer/vulkan/AllocatorHelperRing.h"
#include "libANGLE/renderer/vulkan/SecondaryCommandBuffer.h"

namespace rx
{
namespace vk
{

void SharedCommandBlockAllocator::resetAllocator()
{
    ASSERT(!mAllocator || !mAllocator->isShared());

    if (mAllocSharedCP)
    {
        mAllocSharedCP->releaseAndUpdate(&mAllocReleaseCP);
        mAllocSharedCP = nullptr;
    }

    ASSERT(!mAllocSharedCP && !mAllocReleaseCP.valid());
}

void SharedCommandBlockAllocator::attachAllocator(SharedCommandMemoryAllocator *allocator)
{
    ASSERT(allocator);
    ASSERT(!mAllocator);
    mAllocator = allocator;
    if (mAllocator->isShared())
    {
        mAllocator->releaseToSharedCP();
    }
}

SharedCommandMemoryAllocator *SharedCommandBlockAllocator::detachAllocator(
    bool isCommandBufferEmpty)
{
    ASSERT(mAllocator);
    if (!isCommandBufferEmpty)
    {
        // Must call reset() after detach from non-empty command buffer (OK to have an empty RP)
        ASSERT(!mAllocSharedCP && !mAllocReleaseCP.valid());
        mAllocSharedCP  = mAllocator->acquireSharedCP();
        mAllocReleaseCP = mAllocator->get().getReleaseCheckPoint();
    }
    SharedCommandMemoryAllocator *result = mAllocator;
    mAllocator                           = nullptr;
    return result;
}

void SharedCommandBlockPool::attachAllocator(SharedCommandMemoryAllocator *source)
{
    ASSERT(source);
    RingBufferAllocator &sourceIn = source->get();

    ASSERT(sourceIn.valid());
    ASSERT(mCommandBuffer->hasEmptyCommands());
    ASSERT(mLastCommandBlock == nullptr);
    ASSERT(mFinishedCommandSize == 0);
    ASSERT(!mAllocator.valid());
    mAllocator = std::move(sourceIn);
    mAllocator.setFragmentReserve(kCommandHeaderSize);
    pushNewCommandBlock(mAllocator.allocate(0));
    mAllocator.setListener(this);
}

void SharedCommandBlockPool::detachAllocator(SharedCommandMemoryAllocator *destination)
{
    ASSERT(destination);
    RingBufferAllocator &destinationOut = destination->get();
    ASSERT(!destinationOut.valid());

    ASSERT(mAllocator.valid());
    mAllocator.setListener(nullptr);
    finishLastCommandBlock();
    if (mFinishedCommandSize == 0)
    {
        mCommandBuffer->clearCommands();
    }
    else
    {
        mAllocator.setFragmentReserve(0);
        (void)mAllocator.allocate(sizeof(kCommandHeaderSize));
    }
    destinationOut = std::move(mAllocator);
}

void SharedCommandBlockPool::pushNewCommandBlock(uint8_t *block)
{
    mLastCommandBlock = block;
    mCommandBuffer->pushToCommands(block);
}

void SharedCommandBlockPool::finishLastCommandBlock()
{
    mFinishedCommandSize = getCommandSize();
    terminateLastCommandBlock();
    mLastCommandBlock = nullptr;
}

void SharedCommandBlockPool::onRingBufferNewFragment()
{
    pushNewCommandBlock(mAllocator.getPointer());
}

void SharedCommandBlockPool::onRingBufferFragmentEnd()
{
    finishLastCommandBlock();
}

void SharedCommandBlockPool::getMemoryUsageStats(size_t *usedMemoryOut,
                                                 size_t *allocatedMemoryOut) const
{
    *usedMemoryOut      = getCommandSize();
    *allocatedMemoryOut = getCommandSize();
}

}  // namespace vk
}  // namespace rx
