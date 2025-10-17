/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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
#include "RemoteTextureProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteTextureMessages.h"
#include "RemoteTextureViewProxy.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebGPUTextureViewDescriptor.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteTextureProxy);

RemoteTextureProxy::RemoteTextureProxy(Ref<RemoteGPUProxy>&& root, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier, bool isCanvasBacking)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_root(WTFMove(root))
    , m_isCanvasBacking(isCanvasBacking)
{
}

RemoteTextureProxy::~RemoteTextureProxy()
{
    auto sendResult = send(Messages::RemoteTexture::Destruct());
    UNUSED_VARIABLE(sendResult);
}

static bool equalDescriptors(const std::optional<WebCore::WebGPU::TextureViewDescriptor>& a, const std::optional<WebCore::WebGPU::TextureViewDescriptor>& b)
{
    if (!a && !b)
        return true;

    if (!a || !b)
        return false;

    return a->format == b->format
        && a->dimension == b->dimension
        && a->aspect == b->aspect
        && a->baseMipLevel == b->baseMipLevel
        && a->mipLevelCount == b->mipLevelCount
        && a->baseArrayLayer == b->baseArrayLayer
        && a->arrayLayerCount == b->arrayLayerCount;
}

RefPtr<WebCore::WebGPU::TextureView> RemoteTextureProxy::createView(const std::optional<WebCore::WebGPU::TextureViewDescriptor>& descriptor)
{
    if (m_isCanvasBacking && m_lastCreatedView && equalDescriptors(descriptor, m_lastCreatedViewDescriptor))
        return m_lastCreatedView;

    std::optional<TextureViewDescriptor> convertedDescriptor;
    Ref convertToBackingContext = m_convertToBackingContext;

    if (descriptor) {
        convertedDescriptor = convertToBackingContext->convertToBacking(*descriptor);
        if (!convertedDescriptor)
            return nullptr;
    }

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteTexture::CreateView(*convertedDescriptor, identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    auto result = RemoteTextureViewProxy::create(*this, convertToBackingContext, identifier);
    result->setLabel(WTFMove(convertedDescriptor->label));
    if (!m_isCanvasBacking)
        return result;

    m_lastCreatedView = WTFMove(result);
    m_lastCreatedViewDescriptor = descriptor;
    return m_lastCreatedView;
}

void RemoteTextureProxy::destroy()
{
    auto sendResult = send(Messages::RemoteTexture::Destroy());
    UNUSED_VARIABLE(sendResult);
}

void RemoteTextureProxy::undestroy()
{
    auto sendResult = send(Messages::RemoteTexture::Undestroy());
    UNUSED_VARIABLE(sendResult);
}

void RemoteTextureProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteTexture::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
