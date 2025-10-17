/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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

#import "BindGroupLayout.h"
#import "Pipeline.h"

#import <wtf/FastMalloc.h>
#import <wtf/HashMap.h>
#import <wtf/HashTraits.h>
#import <wtf/Ref.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakPtr.h>

struct WGPUComputePipelineImpl {
};

namespace WebGPU {

class BindGroupLayout;
class Device;
class PipelineLayout;

// https://gpuweb.github.io/gpuweb/#gpucomputepipeline
class ComputePipeline : public RefCountedAndCanMakeWeakPtr<ComputePipeline>, public WGPUComputePipelineImpl {
    WTF_MAKE_TZONE_ALLOCATED(ComputePipeline);
public:
    static Ref<ComputePipeline> create(id<MTLComputePipelineState> computePipelineState, Ref<PipelineLayout>&& pipelineLayout, MTLSize threadsPerThreadgroup, BufferBindingSizesForPipeline&& minimumBufferSizes, Device& device)
    {
        return adoptRef(*new ComputePipeline(computePipelineState, WTFMove(pipelineLayout), threadsPerThreadgroup, WTFMove(minimumBufferSizes), device));
    }
    static Ref<ComputePipeline> createInvalid(Device& device)
    {
        return adoptRef(*new ComputePipeline(device));
    }

    ~ComputePipeline();

    Ref<BindGroupLayout> getBindGroupLayout(uint32_t groupIndex);
    void setLabel(String&&);

    bool isValid() const { return m_computePipelineState; }

    id<MTLComputePipelineState> computePipelineState() const { return m_computePipelineState; }

    Device& device() const { return m_device; }
    MTLSize threadsPerThreadgroup() const { return m_threadsPerThreadgroup; }

    PipelineLayout& pipelineLayout() const { return m_pipelineLayout; }
    Ref<PipelineLayout> protectedPipelineLayout() const { return m_pipelineLayout; }

    const BufferBindingSizesForBindGroup* minimumBufferSizes(uint32_t) const;

private:
    ComputePipeline(id<MTLComputePipelineState>, Ref<PipelineLayout>&&, MTLSize, BufferBindingSizesForPipeline&&, Device&);
    ComputePipeline(Device&);

    const id<MTLComputePipelineState> m_computePipelineState { nil };

    const Ref<Device> m_device;
    const MTLSize m_threadsPerThreadgroup;
    const Ref<PipelineLayout> m_pipelineLayout;
    const BufferBindingSizesForPipeline m_minimumBufferSizes;
};

} // namespace WebGPU
