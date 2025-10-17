/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#include "RemoteXRSubImageProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteGPUProxy.h"
#include "RemoteTextureProxy.h"
#include "RemoteXRSubImageMessages.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/ImageBuffer.h>
#include <WebCore/WebGPUTextureFormat.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteXRSubImageProxy);

RemoteXRSubImageProxy::RemoteXRSubImageProxy(Ref<RemoteGPUProxy>&& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(WTFMove(parent))
{
}

RemoteXRSubImageProxy::~RemoteXRSubImageProxy()
{
    auto sendResult = send(Messages::RemoteXRSubImage::Destruct());
    UNUSED_VARIABLE(sendResult);
}

RefPtr<WebCore::WebGPU::Texture> RemoteXRSubImageProxy::colorTexture()
{
    if (m_currentTexture)
        return m_currentTexture;

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteXRSubImage::GetColorTexture(identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    m_currentTexture = RemoteTextureProxy::create(protectedRoot(), m_convertToBackingContext, identifier);
    return m_currentTexture;
}

RefPtr<WebCore::WebGPU::Texture> RemoteXRSubImageProxy::depthStencilTexture()
{
    if (m_currentDepthTexture)
        return m_currentDepthTexture;

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteXRSubImage::GetDepthTexture(identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    m_currentDepthTexture = RemoteTextureProxy::create(protectedRoot(), m_convertToBackingContext, identifier);
    return m_currentDepthTexture;
}

RefPtr<WebCore::WebGPU::Texture> RemoteXRSubImageProxy::motionVectorTexture()
{
    return nullptr;
}


} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
