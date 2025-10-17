/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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
#include "OpenXRLayer.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WEBXR) && USE(OPENXR)

using namespace WebCore;

namespace PlatformXR {

WTF_MAKE_TZONE_ALLOCATED_IMPL(OpenXRLayer);
WTF_MAKE_TZONE_ALLOCATED_IMPL(OpenXRLayerProjection);

// OpenXRLayerProjection

std::unique_ptr<OpenXRLayerProjection> OpenXRLayerProjection::create(XrInstance instance, XrSession session, uint32_t width, uint32_t height, int64_t format, uint32_t sampleCount)
{
    if (!width || !height || !sampleCount)
        return nullptr;

    auto info = createStructure<XrSwapchainCreateInfo, XR_TYPE_SWAPCHAIN_CREATE_INFO>();
    info.arraySize = 1;
    info.format = format;
    info.width = width;
    info.height = height;
    info.mipCount = 1;
    info.faceCount = 1;
    info.arraySize = 1;
    info.sampleCount = sampleCount;
    info.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;

    auto swapchain = OpenXRSwapchain::create(instance, session, info);
    if (!swapchain)
        return nullptr;

    return std::unique_ptr<OpenXRLayerProjection>(new OpenXRLayerProjection(makeUniqueRefFromNonNullUniquePtr(WTFMove(swapchain))));
}

OpenXRLayerProjection::OpenXRLayerProjection(UniqueRef<OpenXRSwapchain>&& swapchain)
    : m_swapchain(WTFMove(swapchain))
    , m_layerProjection(createStructure<XrCompositionLayerProjection, XR_TYPE_COMPOSITION_LAYER_PROJECTION>())
{
}

std::optional<FrameData::LayerData> OpenXRLayerProjection::startFrame()
{
    auto texture = m_swapchain->acquireImage();
    if (!texture)
        return std::nullopt;

    return FrameData::LayerData {
        .framebufferSize = m_swapchain->size(),
        .opaqueTexture = *texture
    };
}

XrCompositionLayerBaseHeader* OpenXRLayerProjection::endFrame(const Device::Layer& layer, XrSpace space, const Vector<XrView>& frameViews)
{
    ASSERT(layer.views.size() == frameViews.size());

    auto viewCount = frameViews.size();
    m_projectionViews.fill(createStructure<XrCompositionLayerProjectionView, XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW>(), viewCount);
    m_layerProjection.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    for (uint32_t i = 0; i < viewCount; ++i) {
        m_projectionViews[i].pose = frameViews[i].pose;
        m_projectionViews[i].fov = frameViews[i].fov;
        m_projectionViews[i].subImage.swapchain = m_swapchain->swapchain();

        auto& viewport = layer.views[i].viewport;

        m_projectionViews[i].subImage.imageRect.offset = { viewport.x(), viewport.y() };
        m_projectionViews[i].subImage.imageRect.extent = { viewport.width(), viewport.height() };
    }

    m_layerProjection.space = space;
    m_layerProjection.viewCount = m_projectionViews.size();
    m_layerProjection.views = m_projectionViews.data();

    m_swapchain->releaseImage();

    return reinterpret_cast<XrCompositionLayerBaseHeader*>(&m_layerProjection);
}


} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
