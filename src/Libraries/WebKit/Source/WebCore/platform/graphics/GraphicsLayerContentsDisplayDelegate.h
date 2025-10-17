/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#pragma once

#include "GraphicsLayer.h"
#include <wtf/RefCounted.h>

#if !USE(CA) && !USE(COORDINATED_GRAPHICS)
#include "PlatformLayer.h"
#endif

namespace WebCore {
class ImageBuffer;
#if USE(CA)
class PlatformCALayer;
#elif USE(COORDINATED_GRAPHICS)
class CoordinatedPlatformLayer;
class CoordinatedPlatformLayerBuffer;
#endif

// Platform specific interface for attaching contents to GraphicsLayer.
// Responsible for creating compositor resources to show the particular contents
// in the platform specific GraphicsLayer.
class WEBCORE_EXPORT GraphicsLayerContentsDisplayDelegate : public RefCounted<GraphicsLayerContentsDisplayDelegate> {
public:
    virtual ~GraphicsLayerContentsDisplayDelegate();

#if USE(CA)
    virtual void prepareToDelegateDisplay(PlatformCALayer&);
    // Must not detach the platform layer backing store.
    virtual void display(PlatformCALayer&) = 0;
    virtual GraphicsLayer::CompositingCoordinatesOrientation orientation() const;
#elif USE(COORDINATED_GRAPHICS)
    virtual void setDisplayBuffer(std::unique_ptr<CoordinatedPlatformLayerBuffer>&&) = 0;
    virtual bool display(CoordinatedPlatformLayer&) = 0;
#else
    virtual PlatformLayer* platformLayer() const = 0;
#endif
};

class GraphicsLayerAsyncContentsDisplayDelegate : public GraphicsLayerContentsDisplayDelegate {
public:
    virtual ~GraphicsLayerAsyncContentsDisplayDelegate() = default;

    virtual bool WEBCORE_EXPORT tryCopyToLayer(ImageBuffer&) = 0;

    virtual bool isGraphicsLayerAsyncContentsDisplayDelegateCocoa() const { return false; }
    virtual bool isGraphicsLayerCARemoteAsyncContentsDisplayDelegate() const { return false; }
};

}
