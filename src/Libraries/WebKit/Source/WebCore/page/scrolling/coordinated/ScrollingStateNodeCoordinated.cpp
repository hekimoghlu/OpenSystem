/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#include "ScrollingStateNode.h"

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "GraphicsLayerCoordinated.h"

namespace WebCore {

void LayerRepresentation::retainPlatformLayer(void* typelessLayer)
{
    if (auto* layer = makePlatformLayerTyped(typelessLayer))
        layer->ref();
}

void LayerRepresentation::releasePlatformLayer(void* typelessLayer)
{
    if (auto* layer = makePlatformLayerTyped(typelessLayer))
        layer->deref();
}

CoordinatedPlatformLayer* LayerRepresentation::makePlatformLayerTyped(void* typelessLayer)
{
    return static_cast<CoordinatedPlatformLayer*>(typelessLayer);
}

void* LayerRepresentation::makePlatformLayerTypeless(CoordinatedPlatformLayer* layer)
{
    return layer;
}

CoordinatedPlatformLayer* LayerRepresentation::platformLayerFromGraphicsLayer(GraphicsLayer& graphicsLayer)
{
    return &downcast<GraphicsLayerCoordinated>(graphicsLayer).coordinatedPlatformLayer();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
