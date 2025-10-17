/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#import "ScrollingStateNode.h"

#import "PlatformLayer.h"

#if ENABLE(ASYNC_SCROLLING)

namespace WebCore {

void LayerRepresentation::retainPlatformLayer(void* typelessLayer)
{
    if (typelessLayer)
        CFRetain(typelessLayer);
}

void LayerRepresentation::releasePlatformLayer(void* typelessLayer)
{
    if (typelessLayer)
        CFRelease(typelessLayer);
}

CALayer *LayerRepresentation::makePlatformLayerTyped(void* typelessLayer)
{
    return (__bridge CALayer *)typelessLayer;
}

void* LayerRepresentation::makePlatformLayerTypeless(CALayer *layer)
{
    return (__bridge void*)layer;
}

CALayer* LayerRepresentation::platformLayerFromGraphicsLayer(GraphicsLayer& graphicsLayer)
{
    return graphicsLayer.platformLayer();
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
