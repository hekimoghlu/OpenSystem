/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUComputePipeline.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPU.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class BindGroupLayoutImpl;
class ConvertToBackingContext;

class ComputePipelineImpl final : public ComputePipeline {
    WTF_MAKE_TZONE_ALLOCATED(ComputePipelineImpl);
public:
    static Ref<ComputePipelineImpl> create(WebGPUPtr<WGPUComputePipeline>&& computePipeline, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new ComputePipelineImpl(WTFMove(computePipeline), convertToBackingContext));
    }

    virtual ~ComputePipelineImpl();

private:
    friend class DowncastConvertToBackingContext;

    ComputePipelineImpl(WebGPUPtr<WGPUComputePipeline>&&, ConvertToBackingContext&);

    ComputePipelineImpl(const ComputePipelineImpl&) = delete;
    ComputePipelineImpl(ComputePipelineImpl&&) = delete;
    ComputePipelineImpl& operator=(const ComputePipelineImpl&) = delete;
    ComputePipelineImpl& operator=(ComputePipelineImpl&&) = delete;

    WGPUComputePipeline backing() const { return m_backing.get(); }

    Ref<BindGroupLayout> getBindGroupLayout(uint32_t index) final;

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPUComputePipeline> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
