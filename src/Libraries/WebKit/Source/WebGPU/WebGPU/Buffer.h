/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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
#import "SwiftCXXThunk.h"
#import "WebGPU.h"
#import "WebGPUExt.h"
#import <Metal/Metal.h>
#import <utility>
#import <wtf/CompletionHandler.h>
#import <wtf/FastMalloc.h>
#import <wtf/HashMap.h>
#import <wtf/Range.h>
#import <wtf/RangeSet.h>
#import <wtf/Ref.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/RetainReleaseSwift.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakHashSet.h>
#import <wtf/WeakPtr.h>

struct WGPUBufferImpl {
};

namespace WebGPU {

class CommandBuffer;
class CommandEncoder;
class Device;

// https://gpuweb.github.io/gpuweb/#gpubuffer
class Buffer : public WGPUBufferImpl, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<Buffer> {
    WTF_MAKE_TZONE_ALLOCATED(Buffer);
public:
    enum class State : uint8_t;
    struct MappingRange {
        size_t beginOffset; // Inclusive
        size_t endOffset; // Exclusive
    };

    static Ref<Buffer> create(id<MTLBuffer> buffer, uint64_t initialSize, WGPUBufferUsageFlags usage, State initialState, MappingRange initialMappingRange, Device& device)
    {
        return adoptRef(*new Buffer(buffer, initialSize, usage, initialState, initialMappingRange, device));
    }
    static Ref<Buffer> createInvalid(Device& device)
    {
        return adoptRef(*new Buffer(device));
    }

    ~Buffer();

    void destroy();
    std::span<uint8_t> getMappedRange(size_t offset, size_t) HAS_SWIFTCXX_THUNK;
    void mapAsync(WGPUMapModeFlags, size_t offset, size_t, CompletionHandler<void(WGPUBufferMapAsyncStatus)>&& callback);
    void unmap();
    void setLabel(String&&);

    bool isValid() const;

    // https://gpuweb.github.io/gpuweb/#buffer-state
    enum class State : uint8_t {
        Mapped,
        MappedAtCreation,
        MappingPending,
        Unmapped,
        Destroyed,
    };

    id<MTLBuffer> buffer() const { return m_buffer; }
    id<MTLBuffer> indirectBuffer() const;
    id<MTLBuffer> indirectIndexedBuffer() const { return m_indirectIndexedBuffer; }
    id<MTLBuffer> indirectIndexedBuffer(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, MTLIndexType, uint32_t firstInstance, id<MTLIndirectCommandBuffer> = nil);
    id<MTLBuffer> indirectIndexedBuffer(uint32_t firstIndex, uint32_t indexCount, uint32_t firstInstance, id<MTLIndirectCommandBuffer> = nil) const;

    uint64_t initialSize() const;
    uint64_t currentSize() const;
    WGPUBufferUsageFlags usage() const { return m_usage; }
    State state() const { return m_state; }

    Device& device() const { return m_device; }
    Ref<Device> protectedDevice() const { return m_device; }
    bool isDestroyed() const { return state() == State::Destroyed; }

    void setCommandEncoder(CommandEncoder&, bool mayModifyBuffer = false) const;
    std::span<uint8_t> getBufferContents();

    bool indirectIndexedBufferRequiresRecomputation(MTLIndexType, NSUInteger indexBufferOffsetInBytes, uint64_t indirectOffset, uint32_t minVertexCount, uint32_t minInstanceCount) const;
    bool indirectBufferRequiresRecomputation(uint64_t indirectOffset, uint32_t minVertexCount, uint32_t minInstanceCount) const;

    void indirectBufferRecomputed(uint64_t indirectOffset, uint32_t minVertexCount, uint32_t minInstanceCount);
    void indirectIndexedBufferRecomputed(MTLIndexType, NSUInteger indexBufferOffsetInBytes, uint64_t indirectOffset, uint32_t minVertexCount, uint32_t minInstanceCount);

    bool canSkipDrawIndexedValidation(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, MTLIndexType, uint32_t firstInstance, id<MTLIndirectCommandBuffer> = nil) const;
    void drawIndexedValidated(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, MTLIndexType, uint32_t firstInstance, id<MTLIndirectCommandBuffer> = nil);
    void skippedDrawIndexedValidation(CommandEncoder&, uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, uint32_t instanceCount, MTLIndexType, uint32_t firstInstance, uint32_t baseVertex, uint32_t minInstanceCount, uint32_t primitiveOffset, id<MTLIndirectCommandBuffer> = nil);
    void skippedDrawIndirectIndexedValidation(CommandEncoder&, Buffer*, MTLIndexType, uint32_t indexBufferOffsetInBytes, uint64_t indirectOffset, uint32_t minVertexCount, uint32_t minInstanceCount, MTLPrimitiveType);
    void skippedDrawIndirectValidation(CommandEncoder&, uint64_t indirectOffset, uint32_t minVertexCount, uint32_t minInstanceCount);

    bool didReadOOB(id<MTLIndirectCommandBuffer> = nil) const;
    void didReadOOB(uint32_t v, id<MTLIndirectCommandBuffer> = nil);

    void indirectBufferInvalidated(CommandEncoder* = nullptr);
    void indirectBufferInvalidated(CommandEncoder&);
#if ENABLE(WEBGPU_SWIFT)
    void copyFrom(const std::span<const uint8_t>, const size_t offset) HAS_SWIFTCXX_THUNK;
#endif

private:
    Buffer(id<MTLBuffer>, uint64_t initialSize, WGPUBufferUsageFlags, State initialState, MappingRange initialMappingRange, Device&);
    Buffer(Device&);


private PUBLIC_IN_WEBGPU_SWIFT:
    bool validateGetMappedRange(size_t offset, size_t rangeSize) const;

private:
    NSString* errorValidatingMapAsync(WGPUMapModeFlags, size_t offset, size_t rangeSize) const;
    bool validateUnmap() const;
    void setState(State);
    void incrementBufferMapCount();
    void decrementBufferMapCount();
    id<MTLBuffer> makeIndexIndirectBuffer();
    uint64_t mapGPUAddress(MTLResourceID, uint32_t firstInstance) const;
    void takeSlowIndexValidationPath(CommandBuffer&, uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, uint32_t instanceCount, MTLIndexType, uint32_t firstInstance, uint32_t baseVertex, uint32_t minInstanceCount, uint32_t primitiveOffset);
    void takeSlowIndirectIndexValidationPath(CommandBuffer&, Buffer&, MTLIndexType, uint32_t indexBufferOffsetInBytes, uint32_t indirectOffset, uint32_t minVertexCount, uint32_t minInstanceCount, MTLPrimitiveType);
    void takeSlowIndirectValidationPath(CommandBuffer&, uint64_t indirectOffset, uint32_t minVertexCount, uint32_t minInstanceCount);

private PUBLIC_IN_WEBGPU_SWIFT:
    id<MTLBuffer> m_buffer { nil };
private:
    id<MTLBuffer> m_indirectBuffer { nil };
    id<MTLBuffer> m_indirectIndexedBuffer { nil };

    // https://gpuweb.github.io/gpuweb/#buffer-interface

    const uint64_t m_initialSize { 0 };
    const WGPUBufferUsageFlags m_usage { 0 };
    State m_state { State::Unmapped };
    // [[mapping]] is unnecessary; we can just use m_device.contents.
    MappingRange m_mappingRange { 0, 0 };
    using MappedRanges = RangeSet<Range<size_t>>;
private PUBLIC_IN_WEBGPU_SWIFT:
    MappedRanges m_mappedRanges;
private:
    WGPUMapModeFlags m_mapMode { WGPUMapMode_None };
    struct IndirectArgsCache {
        uint64_t indirectOffset { UINT64_MAX };
        uint64_t indexBufferOffsetInBytes { UINT64_MAX };
        uint32_t minVertexCount { 0 };
        uint32_t minInstanceCount { 0 };
        MTLIndexType indexType { MTLIndexTypeUInt16 };
    } m_indirectCache;

    using DrawIndexCacheContainer = HashMap<uint64_t, uint64_t, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>>;
    HashMap<uint64_t, DrawIndexCacheContainer, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> m_drawIndexedCache;

    const Ref<Device> m_device;
    mutable WeakHashSet<CommandEncoder> m_commandEncoders;
    mutable HashMap<uint64_t, uint32_t, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> m_gpuResourceMap;
    WeakHashSet<CommandEncoder> m_skippedValidationCommandEncoders;
    bool m_mustTakeSlowIndexValidationPath { false };
#if CPU(X86_64)
    bool m_mappedAtCreation { false };
#endif
    HashMap<uint64_t, bool, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> m_didReadOOB;
} SWIFT_SHARED_REFERENCE(refBuffer, derefBuffer);

} // namespace WebGPU

inline void refBuffer(WebGPU::Buffer* obj)
{
    WTF::ref(obj);
}

inline void derefBuffer(WebGPU::Buffer* obj)
{
    WTF::deref(obj);
}
