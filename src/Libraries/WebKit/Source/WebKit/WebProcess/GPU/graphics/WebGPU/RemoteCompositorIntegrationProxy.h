/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#if ENABLE(GPU_PROCESS)

#include "RemoteGPUProxy.h"
#include "RemotePresentationContextProxy.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUCompositorIntegration.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class ImageBuffer;
class NativeImage;
namespace WebGPU {
class Device;
}
}

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteCompositorIntegrationProxy final : public WebCore::WebGPU::CompositorIntegration {
    WTF_MAKE_TZONE_ALLOCATED(RemoteCompositorIntegrationProxy);
public:
    static Ref<RemoteCompositorIntegrationProxy> create(RemoteGPUProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteCompositorIntegrationProxy(parent, convertToBackingContext, identifier));
    }

    virtual ~RemoteCompositorIntegrationProxy();

    RemoteGPUProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent; }

    void setPresentationContext(RemotePresentationContextProxy& presentationContext)
    {
        ASSERT(!m_presentationContext);
        m_presentationContext = &presentationContext;
    }

    void paintCompositedResultsToCanvas(WebCore::ImageBuffer&, uint32_t) final;
    void withDisplayBufferAsNativeImage(uint32_t, Function<void(WebCore::NativeImage*)>) final;

private:
    friend class DowncastConvertToBackingContext;

    RemoteCompositorIntegrationProxy(RemoteGPUProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteCompositorIntegrationProxy(const RemoteCompositorIntegrationProxy&) = delete;
    RemoteCompositorIntegrationProxy(RemoteCompositorIntegrationProxy&&) = delete;
    RemoteCompositorIntegrationProxy& operator=(const RemoteCompositorIntegrationProxy&) = delete;
    RemoteCompositorIntegrationProxy& operator=(RemoteCompositorIntegrationProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    
    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }
    template<typename T>
    WARN_UNUSED_RETURN IPC::Connection::SendSyncResult<T> sendSync(T&& message)
    {
        return root().protectedStreamClientConnection()->sendSync(WTFMove(message), backing());
    }

#if PLATFORM(COCOA)
    Vector<MachSendRight> recreateRenderBuffers(int width, int height, WebCore::DestinationColorSpace&&, WebCore::AlphaPremultiplication, WebCore::WebGPU::TextureFormat, WebCore::WebGPU::Device&) override;
#endif

    void prepareForDisplay(uint32_t frameIndex, CompletionHandler<void()>&&) override;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_parent;
    RefPtr<RemotePresentationContextProxy> m_presentationContext;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
