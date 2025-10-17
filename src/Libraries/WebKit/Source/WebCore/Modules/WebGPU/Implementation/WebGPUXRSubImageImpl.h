/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#include "WebGPUXRSubImage.h"

#include <WebGPU/WebGPU.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class XRSubImageImpl final : public XRSubImage {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Ref<XRSubImageImpl> create(WebGPUPtr<WGPUXRSubImage>&& backing, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new XRSubImageImpl(WTFMove(backing), convertToBackingContext));
    }

    virtual ~XRSubImageImpl();

private:
    friend class DowncastConvertToBackingContext;

    explicit XRSubImageImpl(WebGPUPtr<WGPUXRSubImage>&&, ConvertToBackingContext&);

    XRSubImageImpl(const XRSubImageImpl&) = delete;
    XRSubImageImpl(XRSubImageImpl&&) = delete;
    XRSubImageImpl& operator=(const XRSubImageImpl&) = delete;
    XRSubImageImpl& operator=(XRSubImageImpl&&) = delete;
    RefPtr<Texture> colorTexture() final;
    RefPtr<Texture> depthStencilTexture() final;
    RefPtr<Texture> motionVectorTexture() final;

    WGPUXRSubImage backing() const { return m_backing.get(); }

    WebGPUPtr<WGPUXRSubImage> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
