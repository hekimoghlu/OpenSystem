/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#include "RemoteCompositorIntegrationProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteCompositorIntegrationMessages.h"
#include "RemoteGPUProxy.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/ImageBuffer.h>
#include <WebCore/WebGPUTextureFormat.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteCompositorIntegrationProxy);

RemoteCompositorIntegrationProxy::RemoteCompositorIntegrationProxy(RemoteGPUProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
{
}

RemoteCompositorIntegrationProxy::~RemoteCompositorIntegrationProxy()
{
    auto sendResult = send(Messages::RemoteCompositorIntegration::Destruct());
    UNUSED_VARIABLE(sendResult);
}

#if PLATFORM(COCOA)
Vector<MachSendRight> RemoteCompositorIntegrationProxy::recreateRenderBuffers(int width, int height, WebCore::DestinationColorSpace&& destinationColorSpace, WebCore::AlphaPremultiplication alphaMode, WebCore::WebGPU::TextureFormat textureFormat, WebCore::WebGPU::Device& device)
{
    RemoteDeviceProxy& proxyDevice = static_cast<RemoteDeviceProxy&>(device);
    auto sendResult = sendSync(Messages::RemoteCompositorIntegration::RecreateRenderBuffers(width, height, WTFMove(destinationColorSpace), alphaMode, textureFormat, proxyDevice.backing()));
    if (!sendResult.succeeded())
        return { };

    auto [renderBuffers] = sendResult.takeReply();
    return WTFMove(renderBuffers);
}
#endif

void RemoteCompositorIntegrationProxy::prepareForDisplay(uint32_t frameIndex, CompletionHandler<void()>&& completionHandler)
{
    auto sendResult = sendSync(Messages::RemoteCompositorIntegration::PrepareForDisplay(frameIndex));
    UNUSED_VARIABLE(sendResult);
    RefPtr { m_presentationContext }->present(frameIndex);

    completionHandler();
}

void RemoteCompositorIntegrationProxy::paintCompositedResultsToCanvas(WebCore::ImageBuffer& buffer, uint32_t bufferIndex)
{
    buffer.flushDrawingContext();
    auto sendResult = sendSync(Messages::RemoteCompositorIntegration::PaintCompositedResultsToCanvas(buffer.renderingResourceIdentifier(), bufferIndex));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCompositorIntegrationProxy::withDisplayBufferAsNativeImage(uint32_t, Function<void(WebCore::NativeImage*)>)
{
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
