/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#pragma once

#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakPtr.h>
#import <wtf/threads/BinarySemaphore.h>

struct WGPUCommandBufferImpl {
};

namespace WebGPU {

class CommandEncoder;
class Device;

// https://gpuweb.github.io/gpuweb/#gpucommandbuffer
class CommandBuffer : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<CommandBuffer>, public WGPUCommandBufferImpl {
    WTF_MAKE_TZONE_ALLOCATED(CommandBuffer);
public:
    static Ref<CommandBuffer> create(id<MTLCommandBuffer> commandBuffer, Device& device, id<MTLSharedEvent> sharedEvent, uint64_t sharedEventSignalValue, Vector<Function<bool(CommandBuffer&)>>&& onCommitHandlers, CommandEncoder& commandEncoder)
    {
        return adoptRef(*new CommandBuffer(commandBuffer, device, sharedEvent, sharedEventSignalValue, WTFMove(onCommitHandlers), commandEncoder));
    }
    static Ref<CommandBuffer> createInvalid(Device& device)
    {
        return adoptRef(*new CommandBuffer(device));
    }

    ~CommandBuffer();

    void setLabel(String&&);

    bool isValid() const { return m_commandBuffer; }

    id<MTLCommandBuffer> commandBuffer() const { return m_commandBuffer; }

    Device& device() const { return m_device; }
    void makeInvalid(NSString*);
    void makeInvalidDueToCommit(NSString*);
    void setBufferMapCount(int bufferMapCount) { m_bufferMapCount = bufferMapCount; }
    int bufferMapCount() const { return m_bufferMapCount; }

    NSString* lastError() const;
    bool waitForCompletion();
    bool preCommitHandler();
    void postCommitHandler();
    void addPostCommitHandler(Function<void(id<MTLCommandBuffer>)>&&);

private:
    CommandBuffer(id<MTLCommandBuffer>, Device&, id<MTLSharedEvent>, uint64_t sharedEventSignalValue, Vector<Function<bool(CommandBuffer&)>>&&, CommandEncoder&);
    CommandBuffer(Device&);
    void retainTimestampsForOneUpdateLoop();

    id<MTLCommandBuffer> m_commandBuffer { nil };
    id<MTLCommandBuffer> m_cachedCommandBuffer { nil };
    int m_bufferMapCount { 0 };

    const Ref<Device> m_device;
    NSString* m_lastErrorString { nil };
    id<MTLSharedEvent> m_sharedEvent { nil };
    Vector<Function<bool(CommandBuffer&)>> m_preCommitHandlers;
    Vector<Function<void(id<MTLCommandBuffer>)>> m_postCommitHandlers;
    const uint64_t m_sharedEventSignalValue { 0 };
    // FIXME: we should not need this semaphore - https://bugs.webkit.org/show_bug.cgi?id=272353
    BinarySemaphore m_commandBufferComplete;
    RefPtr<CommandEncoder> m_commandEncoder;
};

} // namespace WebGPU
