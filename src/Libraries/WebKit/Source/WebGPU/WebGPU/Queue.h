/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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

#import "Instance.h"
#import <Metal/Metal.h>
#import <wtf/CompletionHandler.h>
#import <wtf/FastMalloc.h>
#import <wtf/HashMap.h>
#import <wtf/Ref.h>
#import <wtf/RetainReleaseSwift.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/ThreadSafeRefCounted.h>
#import <wtf/Vector.h>
#import <wtf/WeakPtr.h>

struct WGPUQueueImpl {
};

namespace WebGPU {

class Buffer;
class CommandBuffer;
class CommandEncoder;
class Device;
class Texture;
class TextureView;

// https://gpuweb.github.io/gpuweb/#gpuqueue
// A device owns its default queue, not the other way around.
class Queue : public WGPUQueueImpl, public ThreadSafeRefCounted<Queue> {
    WTF_MAKE_TZONE_ALLOCATED(Queue);
public:
    static Ref<Queue> create(id<MTLCommandQueue> commandQueue, Adapter& adapter, Device& device)
    {
        return adoptRef(*new Queue(commandQueue, adapter, device));
    }
    static Ref<Queue> createInvalid(Adapter& adapter, Device& device)
    {
        return adoptRef(*new Queue(adapter, device));
    }

    ~Queue();

    void onSubmittedWorkDone(CompletionHandler<void(WGPUQueueWorkDoneStatus)>&& callback);
    void submit(Vector<Ref<WebGPU::CommandBuffer>>&& commands);
    void writeBuffer(Buffer&, uint64_t bufferOffset, std::span<uint8_t> data);
    void writeBuffer(id<MTLBuffer>, uint64_t bufferOffset, std::span<uint8_t> data);
    void clearBuffer(id<MTLBuffer>, NSUInteger offset = 0, NSUInteger size = NSUIntegerMax);
    void writeTexture(const WGPUImageCopyTexture& destination, std::span<uint8_t> data, const WGPUTextureDataLayout&, const WGPUExtent3D& writeSize, bool skipValidation = false);
    void setLabel(String&&);

    void onSubmittedWorkScheduled(Function<void()>&&);

    bool isValid() const { return m_commandQueue; }
    void makeInvalid();
    void setCommittedSignalEvent(id<MTLSharedEvent>, size_t frameIndex);

    const Device& device() const SWIFT_RETURNS_INDEPENDENT_VALUE;
    void clearTextureIfNeeded(const WGPUImageCopyTexture&, NSUInteger);
    id<MTLCommandBuffer> commandBufferWithDescriptor(MTLCommandBufferDescriptor*);
    void commitMTLCommandBuffer(id<MTLCommandBuffer>);
    void removeMTLCommandBuffer(id<MTLCommandBuffer>);
    void setEncoderForBuffer(id<MTLCommandBuffer>, id<MTLCommandEncoder>);
    id<MTLCommandEncoder> encoderForBuffer(id<MTLCommandBuffer>) const;
    void clearTextureViewIfNeeded(TextureView&);
    static bool writeWillCompletelyClear(WGPUTextureDimension, uint32_t widthForMetal, uint32_t logicalSizeWidth, uint32_t heightForMetal, uint32_t logicalSizeHeight, uint32_t depthForMetal, uint32_t logicalSizeDepthOrArrayLayers);
    void endEncoding(id<MTLCommandEncoder>, id<MTLCommandBuffer>) const;

    id<MTLBlitCommandEncoder> ensureBlitCommandEncoder();
    void finalizeBlitCommandEncoder();

    // This can be called on a background thread.
    void scheduleWork(Instance::WorkItem&&);
    uint64_t WARN_UNUSED_RETURN retainCounterSampleBuffer(CommandEncoder&);
    void releaseCounterSampleBuffer(uint64_t);
    void retainTimestampsForOneUpdate(NSMutableSet<id<MTLCounterSampleBuffer>> *);
    void waitForAllCommitedWorkToComplete();
    void synchronizeResourceAndWait(id<MTLBuffer>);
private:
    Queue(id<MTLCommandQueue>, Adapter&, Device&);
    Queue(Adapter&, Device&);

    NSString* errorValidatingSubmit(const Vector<Ref<WebGPU::CommandBuffer>>&) const;
    bool validateWriteBuffer(const Buffer&, uint64_t bufferOffset, size_t) const;


    bool isIdle() const;
    bool isSchedulingIdle() const { return m_submittedCommandBufferCount == m_scheduledCommandBufferCount; }
    void removeMTLCommandBufferInternal(id<MTLCommandBuffer>);

    NSString* errorValidatingWriteTexture(const WGPUImageCopyTexture&, const WGPUTextureDataLayout&, const WGPUExtent3D&, size_t, const Texture&) const;

    id<MTLCommandQueue> m_commandQueue { nil };
    id<MTLCommandBuffer> m_commandBuffer { nil };
    id<MTLBlitCommandEncoder> m_blitCommandEncoder { nil };
private PUBLIC_IN_WEBGPU_SWIFT:
    ThreadSafeWeakPtr<Device> m_device; // The only kind of queues that exist right now are default queues, which are owned by Devices.
private:
    uint64_t m_submittedCommandBufferCount { 0 };
    uint64_t m_completedCommandBufferCount { 0 };
    uint64_t m_scheduledCommandBufferCount { 0 };
    using OnSubmittedWorkScheduledCallbacks = Vector<WTF::Function<void()>>;
    HashMap<uint64_t, OnSubmittedWorkScheduledCallbacks, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> m_onSubmittedWorkScheduledCallbacks;
    using OnSubmittedWorkDoneCallbacks = Vector<WTF::Function<void(WGPUQueueWorkDoneStatus)>>;
    HashMap<uint64_t, OnSubmittedWorkDoneCallbacks, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> m_onSubmittedWorkDoneCallbacks;
    NSMutableDictionary<NSNumber*, NSMutableSet<id<MTLCounterSampleBuffer>>*>* m_retainedCounterSampleBuffers;
    NSMutableOrderedSet<id<MTLCommandBuffer>> *m_createdNotCommittedBuffers { nil };
    NSMutableOrderedSet<id<MTLCommandBuffer>> *m_committedNotCompletedBuffers WTF_GUARDED_BY_LOCK(m_committedNotCompletedBuffersLock) { nil };
    Lock m_committedNotCompletedBuffersLock;
    NSMapTable<id<MTLCommandBuffer>, id<MTLCommandEncoder>> *m_openCommandEncoders;
    const ThreadSafeWeakPtr<Instance> m_instance;
} SWIFT_SHARED_REFERENCE(refQueue, derefQueue);

} // namespace WebGPU

inline void refQueue(WebGPU::Queue* obj)
{
    WTF::ref(obj);
}

inline void derefQueue(WebGPU::Queue* obj)
{
    WTF::deref(obj);
}

