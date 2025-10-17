/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
#include "CoordinatedAnimatedBackingStoreClient.h"

#if USE(COORDINATED_GRAPHICS)
#include "FloatQuad.h"
#include "GraphicsLayerCoordinated.h"
#include "TransformationMatrix.h"
#include <wtf/MainThread.h>

namespace WebCore {

Ref<CoordinatedAnimatedBackingStoreClient> CoordinatedAnimatedBackingStoreClient::create(GraphicsLayer& layer)
{
    return adoptRef(*new CoordinatedAnimatedBackingStoreClient(layer));
}

CoordinatedAnimatedBackingStoreClient::CoordinatedAnimatedBackingStoreClient(GraphicsLayer& layer)
    : m_layer(&layer)
{
}

void CoordinatedAnimatedBackingStoreClient::invalidate()
{
    ASSERT(isMainThread());
    m_layer = nullptr;
}

void CoordinatedAnimatedBackingStoreClient::update(const FloatRect& visibleRect, const FloatRect& coverRect, const FloatSize& size, float contentsScale)
{
    ASSERT(isMainThread());
    m_visibleRect = visibleRect;
    m_coverRect = coverRect;
    m_size = size;
    m_contentsScale = contentsScale;
}

void CoordinatedAnimatedBackingStoreClient::requestBackingStoreUpdateIfNeeded(const TransformationMatrix& transform)
{
    // This is called from the compositor thread.
    ASSERT(!isMainThread());

    // Calculate the contents rectangle of the layer in backingStore coordinates.
    FloatRect contentsRect = { { }, m_size };
    contentsRect.scale(m_contentsScale);

    // If the area covered by tiles (the coverRect, already in backingStore coordinates) covers the whole
    // layer contents then we don't need to do anything.
    if (m_coverRect.contains(contentsRect))
        return;

    // Non-invertible layers are not visible.
    if (!transform.isInvertible())
        return;

    // Calculate the inverse of the layer transformation. The inverse transform will have the inverse of the
    // scaleFactor applied, so we need to scale it back.
    TransformationMatrix inverse = transform.inverse().value_or(TransformationMatrix()).scale(m_contentsScale);

    // Apply the inverse transform to the visible rectangle, so we have the visible rectangle in layer coordinates.
    FloatRect rect = inverse.clampedBoundsOfProjectedQuad(FloatQuad(m_visibleRect));
    GraphicsLayerCoordinated::clampToSizeIfRectIsInfinite(rect, m_size);
    FloatRect transformedVisibleRect = enclosingIntRect(rect);

    // Convert the calculated visible rectangle to backingStore coordinates.
    transformedVisibleRect.scale(m_contentsScale);

    // Restrict the calculated visible rect to the contents rectangle of the layer.
    transformedVisibleRect.intersect(contentsRect);

    if (m_coverRect.contains(transformedVisibleRect))
        return;

    // The coverRect doesn't contain the calculated visible rectangle we need to request a backingStore
    // update to render more tiles.
    callOnMainThread([this, protectedThis = Ref { *this }]() {
        if (m_layer)
            m_layer->client().notifyFlushRequired(m_layer);
    });
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
