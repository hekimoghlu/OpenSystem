/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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

#include "WebGPUIntegralTypes.h"
#include "WebGPUPresentationContext.h"
#include "WebGPUPtr.h"
#include "WebGPUTextureFormat.h"
#include <IOSurface/IOSurfaceRef.h>
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;
class TextureImpl;

class PresentationContextImpl final : public PresentationContext {
    WTF_MAKE_TZONE_ALLOCATED(PresentationContextImpl);
public:
    static Ref<PresentationContextImpl> create(WebGPUPtr<WGPUSurface>&& surface, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new PresentationContextImpl(WTFMove(surface), convertToBackingContext));
    }

    virtual ~PresentationContextImpl();

    void setSize(uint32_t width, uint32_t height);

    void present(uint32_t frameIndex, bool = false);

    WGPUSurface backing() const { return m_backing.get(); }
    RefPtr<WebCore::NativeImage> getMetalTextureAsNativeImage(uint32_t bufferIndex, bool& isIOSurfaceSupportedFormat) final;

private:
    friend class DowncastConvertToBackingContext;

    PresentationContextImpl(WebGPUPtr<WGPUSurface>&&, ConvertToBackingContext&);

    PresentationContextImpl(const PresentationContextImpl&) = delete;
    PresentationContextImpl(PresentationContextImpl&&) = delete;
    PresentationContextImpl& operator=(const PresentationContextImpl&) = delete;
    PresentationContextImpl& operator=(PresentationContextImpl&&) = delete;

    bool configure(const CanvasConfiguration&) final;
    void unconfigure() final;

    RefPtr<Texture> getCurrentTexture(uint32_t) final;

    TextureFormat m_format { TextureFormat::Bgra8unorm };
    uint32_t m_width { 0 };
    uint32_t m_height { 0 };

    WebGPUPtr<WGPUSurface> m_backing;
    WebGPUPtr<WGPUSwapChain> m_swapChain;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    RefPtr<TextureImpl> m_currentTexture;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
