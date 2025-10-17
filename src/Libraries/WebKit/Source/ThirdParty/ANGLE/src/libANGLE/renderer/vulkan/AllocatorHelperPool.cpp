/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
// AllocatorHelperPool:
//    Implements the pool allocator helpers used in the command buffers.
//

#include "libANGLE/renderer/vulkan/AllocatorHelperPool.h"
#include "libANGLE/renderer/vulkan/SecondaryCommandBuffer.h"

namespace rx
{
namespace vk
{

void DedicatedCommandBlockAllocator::resetAllocator()
{
    mAllocator.pop();
    mAllocator.push();
}

void DedicatedCommandBlockPool::reset(CommandBufferCommandTracker *commandBufferTracker)
{
    mCommandBuffer->clearCommands();
    mCurrentWritePointer   = nullptr;
    mCurrentBytesRemaining = 0;
    commandBufferTracker->reset();
}

// Initialize the SecondaryCommandBuffer by setting the allocator it will use
angle::Result DedicatedCommandBlockPool::initialize(DedicatedCommandMemoryAllocator *allocator)
{
    ASSERT(allocator);
    ASSERT(mCommandBuffer->hasEmptyCommands());
    mAllocator = allocator;
    allocateNewBlock();
    // Set first command to Invalid to start
    reinterpret_cast<CommandHeaderIDType &>(*mCurrentWritePointer) = 0;

    return angle::Result::Continue;
}

bool DedicatedCommandBlockPool::empty() const
{
    return mCommandBuffer->checkEmptyForPoolAlloc();
}

void DedicatedCommandBlockPool::allocateNewBlock(size_t blockSize)
{
    ASSERT(mAllocator);
    mCurrentWritePointer   = mAllocator->fastAllocate(blockSize);
    mCurrentBytesRemaining = blockSize;
    mCommandBuffer->pushToCommands(mCurrentWritePointer);
}

void DedicatedCommandBlockPool::getMemoryUsageStats(size_t *usedMemoryOut,
                                                    size_t *allocatedMemoryOut) const
{
    mCommandBuffer->getMemoryUsageStatsForPoolAlloc(kBlockSize, usedMemoryOut, allocatedMemoryOut);
}

}  // namespace vk
}  // namespace rx
