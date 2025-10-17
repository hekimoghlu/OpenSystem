/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "config.h"
#include "WebGPUComputePipelineImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUBindGroupLayoutImpl.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ComputePipelineImpl);

ComputePipelineImpl::ComputePipelineImpl(WebGPUPtr<WGPUComputePipeline>&& computePipeline, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(computePipeline))
    , m_convertToBackingContext(convertToBackingContext)
{
}

ComputePipelineImpl::~ComputePipelineImpl() = default;

Ref<BindGroupLayout> ComputePipelineImpl::getBindGroupLayout(uint32_t index)
{
    // "A new GPUBindGroupLayout wrapper is returned each time"
    return BindGroupLayoutImpl::create(adoptWebGPU(wgpuComputePipelineGetBindGroupLayout(m_backing.get(), index)), m_convertToBackingContext);
}

void ComputePipelineImpl::setLabelInternal(const String& label)
{
    wgpuComputePipelineSetLabel(m_backing.get(), label.utf8().data());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
