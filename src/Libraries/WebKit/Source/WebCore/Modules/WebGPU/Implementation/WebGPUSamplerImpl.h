/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include "WebGPUSampler.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class SamplerImpl final : public Sampler {
    WTF_MAKE_TZONE_ALLOCATED(SamplerImpl);
public:
    static Ref<SamplerImpl> create(WebGPUPtr<WGPUSampler>&& sampler, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new SamplerImpl(WTFMove(sampler), convertToBackingContext));
    }

    virtual ~SamplerImpl();

private:
    friend class DowncastConvertToBackingContext;

    SamplerImpl(WebGPUPtr<WGPUSampler>&&, ConvertToBackingContext&);

    SamplerImpl(const SamplerImpl&) = delete;
    SamplerImpl(SamplerImpl&&) = delete;
    SamplerImpl& operator=(const SamplerImpl&) = delete;
    SamplerImpl& operator=(SamplerImpl&&) = delete;

    WGPUSampler backing() const { return m_backing.get(); }

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPUSampler> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
