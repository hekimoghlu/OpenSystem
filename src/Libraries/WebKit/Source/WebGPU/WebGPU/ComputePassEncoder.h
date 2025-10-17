/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

#import "CommandsMixin.h"
#import "WebGPU.h"
#import "WebGPUExt.h"
#import <wtf/FastMalloc.h>
#import <wtf/HashMap.h>
#import <wtf/Ref.h>
#import <wtf/RefCounted.h>
#import <wtf/RetainReleaseSwift.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>
#import <wtf/WeakPtr.h>

struct WGPUComputePassEncoderImpl {
};

namespace WebGPU {

class BindGroup;
class Buffer;
class CommandEncoder;
class ComputePipeline;
class Device;
class QuerySet;

struct BindableResources;

// https://gpuweb.github.io/gpuweb/#gpucomputepassencoder
class ComputePassEncoder : public WGPUComputePassEncoderImpl, public RefCounted<ComputePassEncoder>, public CommandsMixin {
    WTF_MAKE_TZONE_ALLOCATED(ComputePassEncoder);
public:
    static Ref<ComputePassEncoder> create(id<MTLComputeCommandEncoder> computeCommandEncoder, const WGPUComputePassDescriptor& descriptor, CommandEncoder& parentEncoder, Device& device)
    {
        return adoptRef(*new ComputePassEncoder(computeCommandEncoder, descriptor, parentEncoder, device));
    }
    static Ref<ComputePassEncoder> createInvalid(CommandEncoder& parentEncoder, Device& device, NSString* errorString)
    {
        return adoptRef(*new ComputePassEncoder(parentEncoder, device, errorString));
    }

    ~ComputePassEncoder();

    void dispatch(uint32_t x, uint32_t y, uint32_t z);
    void dispatchIndirect(const Buffer& indirectBuffer, uint64_t indirectOffset);
    void endPass();
    void insertDebugMarker(String&& markerLabel);
    void popDebugGroup();
    void pushDebugGroup(String&& groupLabel);

    void setBindGroup(uint32_t groupIndex, const BindGroup&, std::span<const uint32_t> dynamicOffsets);
    void setPipeline(const ComputePipeline&);
    void setLabel(String&&);

    Device& device() const { return m_device; }

    bool isValid() const;
    id<MTLComputeCommandEncoder> computeCommandEncoder() const;

private:
    ComputePassEncoder(id<MTLComputeCommandEncoder>, const WGPUComputePassDescriptor&, CommandEncoder&, Device&);
    ComputePassEncoder(CommandEncoder&, Device&, NSString*);

    bool validatePopDebugGroup() const;

    void makeInvalid(NSString* = nil);
    void executePreDispatchCommands(const Buffer* = nullptr);
    id<MTLBuffer> runPredispatchIndirectCallValidation(const Buffer&, uint64_t);

    Ref<CommandEncoder> protectedParentEncoder() { return m_parentEncoder; }
    Ref<Device> protectedDevice() const { return m_device; }

    id<MTLComputeCommandEncoder> m_computeCommandEncoder { nil };

    uint64_t m_debugGroupStackSize { 0 };

    const Ref<Device> m_device;
    MTLSize m_threadsPerThreadgroup;
    Vector<uint32_t> m_computeDynamicOffsets;
    Vector<uint32_t> m_priorComputeDynamicOffsets;
    RefPtr<const ComputePipeline> m_pipeline;
    Ref<CommandEncoder> m_parentEncoder;
    HashMap<uint32_t, Vector<uint32_t>, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> m_bindGroupDynamicOffsets;
    HashMap<uint32_t, Vector<const BindableResources*>, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> m_bindGroupResources;
    HashMap<uint32_t, RefPtr<const BindGroup>, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> m_bindGroups;
    NSString *m_lastErrorString { nil };
    bool m_passEnded { false };
} SWIFT_SHARED_REFERENCE(refComputePassEncoder, derefComputePassEncoder);


} // namespace WebGPU

inline void refComputePassEncoder(WebGPU::ComputePassEncoder* obj)
{
    ref(obj);
}

inline void derefComputePassEncoder(WebGPU::ComputePassEncoder* obj)
{
    deref(obj);
}
