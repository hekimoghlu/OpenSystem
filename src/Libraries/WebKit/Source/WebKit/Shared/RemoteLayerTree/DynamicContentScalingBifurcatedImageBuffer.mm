/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#import "config.h"
#import "DynamicContentScalingBifurcatedImageBuffer.h"

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)

#import "DynamicContentScalingImageBufferBackend.h"
#import <CoreRE/RECGCommandsContext.h>
#import <WebCore/BifurcatedGraphicsContext.h>
#import <WebCore/DynamicContentScalingDisplayList.h>
#import <wtf/MachSendRight.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DynamicContentScalingBifurcatedImageBuffer);

DynamicContentScalingBifurcatedImageBuffer::DynamicContentScalingBifurcatedImageBuffer(Parameters parameters, const WebCore::ImageBufferBackend::Info& backendInfo, const WebCore::ImageBufferCreationContext& creationContext, std::unique_ptr<WebCore::ImageBufferBackend>&& backend, WebCore::RenderingResourceIdentifier renderingResourceIdentifier)
    : ImageBuffer(parameters, backendInfo, creationContext, WTFMove(backend), renderingResourceIdentifier)
    , m_dynamicContentScalingBackend(DynamicContentScalingImageBufferBackend::create(ImageBuffer::backendParameters(parameters), creationContext))
{
}

WebCore::GraphicsContext& DynamicContentScalingBifurcatedImageBuffer::context() const
{
    if (!m_context)
        m_context = makeUnique<WebCore::BifurcatedGraphicsContext>(m_backend->context(), m_dynamicContentScalingBackend->context());
    return *m_context;
}

std::optional<WebCore::DynamicContentScalingDisplayList> DynamicContentScalingBifurcatedImageBuffer::dynamicContentScalingDisplayList()
{
    if (!m_dynamicContentScalingBackend)
        return std::nullopt;
    auto* sharing = static_cast<WebCore::ImageBufferBackend&>(*m_dynamicContentScalingBackend).toBackendSharing();
    auto* imageSharing = dynamicDowncast<ImageBufferBackendHandleSharing>(sharing);
    if (!imageSharing)
        return std::nullopt;
    auto handle = imageSharing->takeBackendHandle();
    if (!handle || !std::holds_alternative<WebCore::DynamicContentScalingDisplayList>(*handle))
        return std::nullopt;
    auto& displayList = std::get<WebCore::DynamicContentScalingDisplayList>(*handle);

    // Avoid accumulating display lists; drop the current context and start fresh.
    releaseGraphicsContext();

    return { WTFMove(displayList) };
}

void DynamicContentScalingBifurcatedImageBuffer::releaseGraphicsContext()
{
    ImageBuffer::releaseGraphicsContext();

    m_dynamicContentScalingBackend->releaseGraphicsContext();
    m_context = nullptr;
}

}

#endif // ENABLE(RE_DYNAMIC_CONTENT_SCALING)
