/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
#include "RemoteXRBindingProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteDeviceProxy.h"
#include "RemoteGPUProxy.h"
#include "RemoteXRBindingMessages.h"
#include "RemoteXRProjectionLayerProxy.h"
#include "RemoteXRSubImageProxy.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/ImageBuffer.h>
#include <WebCore/WebGPUTextureFormat.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteXRBindingProxy);

RemoteXRBindingProxy::RemoteXRBindingProxy(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
{
}

RemoteXRBindingProxy::~RemoteXRBindingProxy()
{
    auto sendResult = send(Messages::RemoteXRBinding::Destruct());
    UNUSED_VARIABLE(sendResult);
}

RefPtr<WebCore::WebGPU::XRProjectionLayer> RemoteXRBindingProxy::createProjectionLayer(const WebCore::WebGPU::XRProjectionLayerInit& descriptor)
{
    auto identifier = WebGPUIdentifier::generate();

    auto sendResult = send(Messages::RemoteXRBinding::CreateProjectionLayer(descriptor.colorFormat, descriptor.depthStencilFormat, descriptor.textureUsage, descriptor.scaleFactor, identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    auto result = RemoteXRProjectionLayerProxy::create(protectedRoot(), m_convertToBackingContext, identifier);
    return result;
}

RefPtr<WebCore::WebGPU::XRSubImage> RemoteXRBindingProxy::getSubImage(WebCore::WebGPU::XRProjectionLayer&, WebCore::WebXRFrame&, std::optional<WebCore::WebGPU::XREye>/* = "none"*/)
{
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

RefPtr<WebCore::WebGPU::XRSubImage> RemoteXRBindingProxy::getViewSubImage(WebCore::WebGPU::XRProjectionLayer& projectionLayer)
{
    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteXRBinding::GetViewSubImage(static_cast<RemoteXRProjectionLayerProxy&>(projectionLayer).backing(), identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    auto result = RemoteXRSubImageProxy::create(protectedRoot(), m_convertToBackingContext, identifier);
    return result;
}

WebCore::WebGPU::TextureFormat RemoteXRBindingProxy::getPreferredColorFormat()
{
    return WebCore::WebGPU::TextureFormat::Bgra8unormSRGB;
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
