/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#include "CoordinatedPlatformLayerBufferHolePunch.h"

#if USE(COORDINATED_GRAPHICS) && ENABLE(VIDEO)
#include "IntRect.h"
#include "TextureMapper.h"

#if USE(GSTREAMER)
#include "GStreamerQuirks.h"
#endif

namespace WebCore {

std::unique_ptr<CoordinatedPlatformLayerBufferHolePunch> CoordinatedPlatformLayerBufferHolePunch::create(const IntSize& size)
{
    return makeUnique<CoordinatedPlatformLayerBufferHolePunch>(size);
}

CoordinatedPlatformLayerBufferHolePunch::CoordinatedPlatformLayerBufferHolePunch(const IntSize& size)
    : CoordinatedPlatformLayerBuffer(Type::HolePunch, size, { TextureMapperFlags::ShouldNotBlend }, nullptr)
{
}

#if USE(GSTREAMER)
std::unique_ptr<CoordinatedPlatformLayerBufferHolePunch> CoordinatedPlatformLayerBufferHolePunch::create(const IntSize& size, GstElement* videoSink, RefPtr<GStreamerQuirksManager>&& quirksManager)
{
    ASSERT(videoSink && quirksManager);
    return makeUnique<CoordinatedPlatformLayerBufferHolePunch>(size, videoSink, WTFMove(quirksManager));
}

CoordinatedPlatformLayerBufferHolePunch::CoordinatedPlatformLayerBufferHolePunch(const IntSize& size, GstElement* videoSink, RefPtr<GStreamerQuirksManager>&& quirksManager)
    : CoordinatedPlatformLayerBuffer(Type::HolePunch, size, { TextureMapperFlags::ShouldNotBlend }, nullptr)
    , m_videoSink(videoSink)
    , m_quirksManager(quirksManager)
{
}
#endif

CoordinatedPlatformLayerBufferHolePunch::~CoordinatedPlatformLayerBufferHolePunch() = default;

void CoordinatedPlatformLayerBufferHolePunch::paintToTextureMapper(TextureMapper& textureMapper, const FloatRect& targetRect, const TransformationMatrix& modelViewMatrix, float)
{
#if USE(GSTREAMER)
    if (m_videoSink && m_quirksManager)
        m_quirksManager->setHolePunchVideoRectangle(m_videoSink.get(), enclosingIntRect(modelViewMatrix.mapRect(targetRect)));
#endif
    textureMapper.drawSolidColor(targetRect, modelViewMatrix, Color::transparentBlack, false);
}

void CoordinatedPlatformLayerBufferHolePunch::notifyVideoPosition(const FloatRect& targetRect, const TransformationMatrix& modelViewMatrix)
{
#if USE(GSTREAMER)
    if (m_videoSink && m_quirksManager)
        m_quirksManager->setHolePunchVideoRectangle(m_videoSink.get(), enclosingIntRect(modelViewMatrix.mapRect(targetRect)));
#else
    UNUSED_PARAM(targetRect);
    UNUSED_PARAM(modelViewMatrix);
#endif
}

void CoordinatedPlatformLayerBufferHolePunch::paintTransparentRectangle(TextureMapper& textureMapper, const FloatRect& targetRect, const TransformationMatrix& modelViewMatrix)
{
    textureMapper.drawSolidColor(targetRect, modelViewMatrix, Color::transparentBlack, false);
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS) && ENABLE(VIDEO)
