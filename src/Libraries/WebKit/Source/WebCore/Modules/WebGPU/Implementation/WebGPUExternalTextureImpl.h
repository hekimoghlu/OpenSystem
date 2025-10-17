/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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

#include "WebGPUExternalTexture.h"
#include "WebGPUExternalTextureDescriptor.h"
#include "WebGPUPredefinedColorSpace.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPU.h>
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;
struct ExternalTextureDescriptor;

class ExternalTextureImpl final : public ExternalTexture {
    WTF_MAKE_TZONE_ALLOCATED(ExternalTextureImpl);
public:
    static Ref<ExternalTextureImpl> create(WebGPUPtr<WGPUExternalTexture>&& externalTexture, const ExternalTextureDescriptor& descriptor, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new ExternalTextureImpl(WTFMove(externalTexture), descriptor, convertToBackingContext));
    }

    virtual ~ExternalTextureImpl();

    WGPUExternalTexture backing() const { return m_backing.get(); };

private:
    friend class DowncastConvertToBackingContext;

    ExternalTextureImpl(WebGPUPtr<WGPUExternalTexture>&&, const ExternalTextureDescriptor&, ConvertToBackingContext&);

    ExternalTextureImpl(const ExternalTextureImpl&) = delete;
    ExternalTextureImpl(ExternalTextureImpl&&) = delete;
    ExternalTextureImpl& operator=(const ExternalTextureImpl&) = delete;
    ExternalTextureImpl& operator=(ExternalTextureImpl&&) = delete;

    void setLabelInternal(const String&) final;
    void destroy() final;
    void undestroy() final;
    void updateExternalTexture(CVPixelBufferRef) final;

    Ref<ConvertToBackingContext> m_convertToBackingContext;

    WebGPUPtr<WGPUExternalTexture> m_backing;
    PredefinedColorSpace m_colorSpace;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
