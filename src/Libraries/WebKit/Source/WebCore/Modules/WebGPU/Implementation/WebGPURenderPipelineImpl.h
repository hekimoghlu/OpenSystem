/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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

#include "WebGPUPtr.h"
#include "WebGPURenderPipeline.h"
#include <WebGPU/WebGPU.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class BindGroupLayoutImpl;
class ConvertToBackingContext;

class RenderPipelineImpl final : public RenderPipeline {
    WTF_MAKE_TZONE_ALLOCATED(RenderPipelineImpl);
public:
    static Ref<RenderPipelineImpl> create(WebGPUPtr<WGPURenderPipeline>&& renderPipeline, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new RenderPipelineImpl(WTFMove(renderPipeline), convertToBackingContext));
    }

    virtual ~RenderPipelineImpl();

private:
    friend class DowncastConvertToBackingContext;

    RenderPipelineImpl(WebGPUPtr<WGPURenderPipeline>&&, ConvertToBackingContext&);

    RenderPipelineImpl(const RenderPipelineImpl&) = delete;
    RenderPipelineImpl(RenderPipelineImpl&&) = delete;
    RenderPipelineImpl& operator=(const RenderPipelineImpl&) = delete;
    RenderPipelineImpl& operator=(RenderPipelineImpl&&) = delete;

    WGPURenderPipeline backing() const { return m_backing.get(); }

    Ref<BindGroupLayout> getBindGroupLayout(uint32_t index) final;

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPURenderPipeline> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
