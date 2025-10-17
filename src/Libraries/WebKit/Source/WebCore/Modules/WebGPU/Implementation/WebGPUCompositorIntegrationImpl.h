/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#include "WebGPUCompositorIntegration.h"

#include "WebGPUPresentationContextImpl.h"
#include <WebCore/IOSurface.h>
#include <WebGPU/WebGPU.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

#if PLATFORM(COCOA)
#include <wtf/MachSendRight.h>
#include <wtf/RetainPtr.h>
#include <wtf/spi/cocoa/IOSurfaceSPI.h>
#endif

namespace WebCore {
class Device;
class NativeImage;
}

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class CompositorIntegrationImpl final : public CompositorIntegration {
    WTF_MAKE_TZONE_ALLOCATED(CompositorIntegrationImpl);
public:
    static Ref<CompositorIntegrationImpl> create(ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new CompositorIntegrationImpl(convertToBackingContext));
    }

    virtual ~CompositorIntegrationImpl();

    void setPresentationContext(PresentationContextImpl& presentationContext)
    {
        ASSERT(!m_presentationContext);
        m_presentationContext = &presentationContext;
    }

    void registerCallbacks(WTF::Function<void(CFArrayRef)>&& renderBuffersWereRecreatedCallback, WTF::Function<void(CompletionHandler<void()>&&)>&& onSubmittedWorkScheduledCallback)
    {
        ASSERT(!m_renderBuffersWereRecreatedCallback);
        m_renderBuffersWereRecreatedCallback = WTFMove(renderBuffersWereRecreatedCallback);
        ASSERT(!m_onSubmittedWorkScheduledCallback);
        m_onSubmittedWorkScheduledCallback = WTFMove(onSubmittedWorkScheduledCallback);
    }

    void withDisplayBufferAsNativeImage(uint32_t bufferIndex, Function<void(WebCore::NativeImage*)>) final;
    void paintCompositedResultsToCanvas(WebCore::ImageBuffer&, uint32_t) final;

private:
    friend class DowncastConvertToBackingContext;

    explicit CompositorIntegrationImpl(ConvertToBackingContext&);

    CompositorIntegrationImpl(const CompositorIntegrationImpl&) = delete;
    CompositorIntegrationImpl(CompositorIntegrationImpl&&) = delete;
    CompositorIntegrationImpl& operator=(const CompositorIntegrationImpl&) = delete;
    CompositorIntegrationImpl& operator=(CompositorIntegrationImpl&&) = delete;

    void prepareForDisplay(uint32_t frameIndex, CompletionHandler<void()>&&) override;

#if PLATFORM(COCOA)
    Vector<MachSendRight> recreateRenderBuffers(int width, int height, WebCore::DestinationColorSpace&&, WebCore::AlphaPremultiplication, WebCore::WebGPU::TextureFormat, Device&) override;

    Vector<UniqueRef<WebCore::IOSurface>> m_renderBuffers;
    WTF::Function<void(CFArrayRef)> m_renderBuffersWereRecreatedCallback;
#endif

    WTF::Function<void(CompletionHandler<void()>&&)> m_onSubmittedWorkScheduledCallback;

    RefPtr<PresentationContextImpl> m_presentationContext;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    WeakPtr<Device> m_device;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
