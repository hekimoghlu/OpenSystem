/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#import "config.h"
#import "CommandBuffer.h"

#import "APIConversions.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CommandBuffer);

CommandBuffer::CommandBuffer(id<MTLCommandBuffer> commandBuffer, Device& device, id<MTLSharedEvent> sharedEvent, uint64_t sharedEventSignalValue, Vector<Function<bool(CommandBuffer&)>>&& onCommitHandlers, CommandEncoder& commandEncoder)
    : m_commandBuffer(commandBuffer)
    , m_device(device)
    , m_sharedEvent(sharedEvent)
    , m_preCommitHandlers(WTFMove(onCommitHandlers))
    , m_sharedEventSignalValue(sharedEventSignalValue)
    , m_commandEncoder(&commandEncoder)
{
}

CommandBuffer::CommandBuffer(Device& device)
    : m_device(device)
{
}

void CommandBuffer::retainTimestampsForOneUpdateLoop()
{
    // Workaround for rdar://143905417
    if (RefPtr commandEncoder = m_commandEncoder)
        m_device->protectedQueue()->retainTimestampsForOneUpdate(commandEncoder->timestampBuffers());
}

CommandBuffer::~CommandBuffer()
{
    retainTimestampsForOneUpdateLoop();
    m_device->protectedQueue()->removeMTLCommandBuffer(m_commandBuffer);
    m_commandBuffer = nil;
    m_cachedCommandBuffer = nil;
}

void CommandBuffer::setLabel(String&& label)
{
    m_commandBuffer.label = label;
}

void CommandBuffer::makeInvalid(NSString* lastError)
{
    if (!m_commandBuffer || m_commandBuffer.status >= MTLCommandBufferStatusCommitted)
        return;

    m_lastErrorString = lastError;
    m_device->protectedQueue()->removeMTLCommandBuffer(m_commandBuffer);
    retainTimestampsForOneUpdateLoop();
    m_commandBuffer = nil;
    m_cachedCommandBuffer = nil;
    m_commandEncoder = nullptr;
    m_preCommitHandlers.clear();
    m_postCommitHandlers.clear();
}

bool CommandBuffer::preCommitHandler()
{
    bool result = true;
    for (auto& function : m_preCommitHandlers)
        result = function(*this) && result;

    m_preCommitHandlers.clear();
    return result;
}

void CommandBuffer::postCommitHandler()
{
    for (auto& function : m_postCommitHandlers)
        function(m_cachedCommandBuffer);

    m_postCommitHandlers.clear();
}

void CommandBuffer::addPostCommitHandler(Function<void(id<MTLCommandBuffer>)>&& function)
{
    m_postCommitHandlers.append(WTFMove(function));
}

void CommandBuffer::makeInvalidDueToCommit(NSString* lastError)
{
    if (m_sharedEvent)
        [m_commandBuffer encodeSignalEvent:m_sharedEvent value:m_sharedEventSignalValue];

    m_cachedCommandBuffer = m_commandBuffer;
    [m_commandBuffer addCompletedHandler:[protectedThis = Ref { *this }](id<MTLCommandBuffer>) {
        protectedThis->m_commandBufferComplete.signal();
        protectedThis->m_device->protectedQueue()->scheduleWork([protectedThis = WTFMove(protectedThis)]() mutable {
            protectedThis->m_cachedCommandBuffer = nil;
            protectedThis->m_commandEncoder = nullptr;
        });
    }];
    m_lastErrorString = lastError;
    m_commandBuffer = nil;
}

NSString* CommandBuffer::lastError() const
{
    return m_lastErrorString;
}

bool CommandBuffer::waitForCompletion()
{
    auto status = [m_cachedCommandBuffer status];
    constexpr auto commandBufferSubmissionTimeout = 500_ms;
    if (status == MTLCommandBufferStatusCommitted || status == MTLCommandBufferStatusScheduled)
        return m_commandBufferComplete.waitFor(commandBufferSubmissionTimeout);

    return true;
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuCommandBufferReference(WGPUCommandBuffer commandBuffer)
{
    WebGPU::fromAPI(commandBuffer).ref();
}

void wgpuCommandBufferRelease(WGPUCommandBuffer commandBuffer)
{
    WebGPU::fromAPI(commandBuffer).deref();
}

void wgpuCommandBufferSetLabel(WGPUCommandBuffer commandBuffer, const char* label)
{
    WebGPU::protectedFromAPI(commandBuffer)->setLabel(WebGPU::fromAPI(label));
}
