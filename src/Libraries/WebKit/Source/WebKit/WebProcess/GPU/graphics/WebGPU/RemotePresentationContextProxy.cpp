/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#include "RemotePresentationContextProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemotePresentationContextMessages.h"
#include "RemoteTextureProxy.h"
#include "WebGPUCanvasConfiguration.h"
#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemotePresentationContextProxy);

RemotePresentationContextProxy::RemotePresentationContextProxy(RemoteGPUProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
{
}

RemotePresentationContextProxy::~RemotePresentationContextProxy() = default;

bool RemotePresentationContextProxy::configure(const WebCore::WebGPU::CanvasConfiguration& canvasConfiguration)
{
    auto convertedConfiguration = protectedConvertToBackingContext()->convertToBacking(canvasConfiguration);
    if (!convertedConfiguration)
        return false;

    auto sendResult = send(Messages::RemotePresentationContext::Configure(*convertedConfiguration));
    return sendResult == IPC::Error::NoError;
}

void RemotePresentationContextProxy::unconfigure()
{
    for (size_t i = 0; i < textureCount; ++i)
        m_currentTexture[i] = nullptr;

    auto sendResult = send(Messages::RemotePresentationContext::Unconfigure());
    UNUSED_VARIABLE(sendResult);
}

RefPtr<WebCore::WebGPU::Texture> RemotePresentationContextProxy::getCurrentTexture(uint32_t frameIndex)
{
    if (frameIndex >= textureCount)
        return nullptr;

    if (!m_currentTexture[frameIndex]) {
        auto identifier = WebGPUIdentifier::generate();
        auto sendResult = send(Messages::RemotePresentationContext::GetCurrentTexture(identifier, frameIndex));
        if (sendResult != IPC::Error::NoError)
            return nullptr;

        m_currentTexture[frameIndex] = RemoteTextureProxy::create(protectedRoot(), protectedConvertToBackingContext(), identifier, true);
    } else
        RefPtr { m_currentTexture[frameIndex] }->undestroy();

    return m_currentTexture[frameIndex];
}

void RemotePresentationContextProxy::present(uint32_t frameIndex, bool presentToGPUProcess)
{
    if (presentToGPUProcess) {
        auto sendResult = send(Messages::RemotePresentationContext::Present(frameIndex));
        UNUSED_VARIABLE(sendResult);
    }
}

RefPtr<WebCore::NativeImage> RemotePresentationContextProxy::getMetalTextureAsNativeImage(uint32_t, bool&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

Ref<ConvertToBackingContext> RemotePresentationContextProxy::protectedConvertToBackingContext() const
{
    return m_convertToBackingContext;
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
