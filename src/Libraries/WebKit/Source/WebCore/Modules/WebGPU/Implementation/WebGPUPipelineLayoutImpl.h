/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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

#include "WebGPUPipelineLayout.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class PipelineLayoutImpl final : public PipelineLayout {
    WTF_MAKE_TZONE_ALLOCATED(PipelineLayoutImpl);
public:
    static Ref<PipelineLayoutImpl> create(WebGPUPtr<WGPUPipelineLayout>&& pipelineLayout, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new PipelineLayoutImpl(WTFMove(pipelineLayout), convertToBackingContext));
    }

    virtual ~PipelineLayoutImpl();

private:
    friend class DowncastConvertToBackingContext;

    PipelineLayoutImpl(WebGPUPtr<WGPUPipelineLayout>&&, ConvertToBackingContext&);

    PipelineLayoutImpl(const PipelineLayoutImpl&) = delete;
    PipelineLayoutImpl(PipelineLayoutImpl&&) = delete;
    PipelineLayoutImpl& operator=(const PipelineLayoutImpl&) = delete;
    PipelineLayoutImpl& operator=(PipelineLayoutImpl&&) = delete;

    WGPUPipelineLayout backing() const { return m_backing.get(); }

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPUPipelineLayout> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
