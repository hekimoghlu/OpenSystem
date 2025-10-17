/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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

namespace WebCore {

class FloatRect;
class GraphicsContext;
class PlatformCALayer;

class PlatformCALayerClient {
public:
    virtual PlatformLayerIdentifier platformCALayerIdentifier() const = 0;

    virtual void platformCALayerLayoutSublayersOfLayer(PlatformCALayer*) { }
    virtual bool platformCALayerRespondsToLayoutChanges() const { return false; }

    virtual void platformCALayerCustomSublayersChanged(PlatformCALayer*) { }

    virtual void platformCALayerAnimationStarted(const String& /*animationKey*/, MonotonicTime) { }
    virtual void platformCALayerAnimationEnded(const String& /*animationKey*/) { }
    virtual GraphicsLayer::CompositingCoordinatesOrientation platformCALayerContentsOrientation() const { return GraphicsLayer::CompositingCoordinatesOrientation::TopDown; }
    virtual void platformCALayerPaintContents(PlatformCALayer*, GraphicsContext&, const FloatRect& inClip, OptionSet<GraphicsLayerPaintBehavior>) = 0;
    virtual bool platformCALayerShowDebugBorders() const { return false; }
    virtual bool platformCALayerShowRepaintCounter(PlatformCALayer*) const { return false; }
    virtual int platformCALayerRepaintCount(PlatformCALayer*) const { return 0; }
    virtual int platformCALayerIncrementRepaintCount(PlatformCALayer*) { return 0; }
    
    virtual bool platformCALayerContentsOpaque() const = 0;
    virtual bool platformCALayerDrawsContent() const = 0;
    virtual bool platformCALayerDelegatesDisplay(PlatformCALayer*) const { return false; };
    virtual void platformCALayerLayerDisplay(PlatformCALayer*) { }
    virtual void platformCALayerLayerDidDisplay(PlatformCALayer*) { }

    virtual bool platformCALayerRenderingIsSuppressedIncludingDescendants() const { return false; }

    virtual void platformCALayerSetNeedsToRevalidateTiles() { }
    virtual float platformCALayerDeviceScaleFactor() const = 0;
    virtual float platformCALayerContentsScaleMultiplierForNewTiles(PlatformCALayer*) const { return 1; }

    virtual bool platformCALayerShouldAggressivelyRetainTiles(PlatformCALayer*) const { return false; }
    virtual bool platformCALayerShouldTemporarilyRetainTileCohorts(PlatformCALayer*) const { return true; }

    virtual bool platformCALayerUseGiantTiles() const { return false; }
    virtual bool platformCALayerCSSUnprefixedBackdropFilterEnabled() const { return false; }

    virtual bool isCommittingChanges() const { return false; }

    virtual bool isUsingDisplayListDrawing(PlatformCALayer*) const { return false; }

#if HAVE(HDR_SUPPORT)
    virtual bool hdrForImagesEnabled() const { return false; }
#endif

    virtual void platformCALayerLogFilledVisibleFreshTile(unsigned /* blankPixelCount */) { }

    virtual bool platformCALayerContainsBitmapOnly(const PlatformCALayer*) const { return false; }

    virtual bool platformCALayerShouldPaintUsingCompositeCopy() const { return false; }

    virtual bool platformCALayerNeedsPlatformContext(const PlatformCALayer*) const { return false; }

protected:
    virtual ~PlatformCALayerClient() = default;
};

}

