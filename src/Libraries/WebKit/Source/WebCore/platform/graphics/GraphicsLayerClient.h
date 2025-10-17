/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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

#include "LayerTreeAsTextOptions.h"
#include "TiledBacking.h"
#include "TransformationMatrix.h"
#include <wtf/Forward.h>
#include <wtf/OptionSet.h>

namespace WebCore {

class FloatPoint;
class FloatRect;
class GraphicsContext;
class GraphicsLayer;
class IntPoint;
class IntRect;

enum class AnimatedProperty : uint8_t {
    Invalid,
    Translate,
    Scale,
    Rotate,
    Transform,
    Opacity,
    BackgroundColor,
    Filter,
    WebkitBackdropFilter,
};

enum class GraphicsLayerPaintingPhase {
    Background            = 1 << 0,
    Foreground            = 1 << 1,
    Mask                  = 1 << 2,
    ClipPath              = 1 << 3,
    OverflowContents      = 1 << 4,
    CompositedScroll      = 1 << 5,
    ChildClippingMask     = 1 << 6,
};

enum class PlatformLayerTreeAsTextFlags : uint8_t {
    Debug = 1 << 0,
    IgnoreChildren = 1 << 1,
    IncludeModels = 1 << 2,
};

// See WebCore::PaintBehavior.
enum class GraphicsLayerPaintBehavior : uint8_t {
    DefaultAsynchronousImageDecode = 1 << 0,
    ForceSynchronousImageDecode = 1 << 1,
};
    
class GraphicsLayerClient {
public:
    virtual ~GraphicsLayerClient() = default;

    virtual void tiledBackingUsageChanged(const GraphicsLayer*, bool /*usingTiledBacking*/) { }
    
    // Callback for when hardware-accelerated animation started.
    virtual void notifyAnimationStarted(const GraphicsLayer*, const String& /*animationKey*/, MonotonicTime /*time*/) { }
    virtual void notifyAnimationEnded(const GraphicsLayer*, const String& /*animationKey*/) { }

    // Notification that a layer property changed that requires a subsequent call to flushCompositingState()
    // to appear on the screen.
    virtual void notifyFlushRequired(const GraphicsLayer*) { }

    // Notification that this layer requires a flush on the next display refresh.
    virtual void notifySubsequentFlushRequired(const GraphicsLayer*) { }

    virtual void paintContents(const GraphicsLayer*, GraphicsContext&, const FloatRect& /* inClip */, OptionSet<GraphicsLayerPaintBehavior>) { }
    virtual void didChangePlatformLayerForLayer(const GraphicsLayer*) { }

    // Provides current transform (taking transform-origin and animations into account). Input matrix has been
    // initialized to identity already. Returns false if the layer has no transform.
    virtual bool getCurrentTransform(const GraphicsLayer*, TransformationMatrix&) const { return false; }

    // Allows the client to modify a layer position used during the visibleRect calculation, for example to ignore
    // scroll overhang.
    virtual void customPositionForVisibleRectComputation(const GraphicsLayer*, FloatPoint&) const { }

    // Multiplier for backing store size, related to high DPI.
    virtual float deviceScaleFactor() const { return 1; }
    // Page scale factor.
    virtual float pageScaleFactor() const { return 1; }
    virtual float zoomedOutPageScaleFactor() const { return 0; }

    virtual std::optional<float> customContentsScale(const GraphicsLayer*) const { return { }; }

    virtual float contentsScaleMultiplierForNewTiles(const GraphicsLayer*) const { return 1; }
    virtual bool paintsOpaquelyAtNonIntegralScales(const GraphicsLayer*) const { return false; }

    virtual bool isFlushingLayers() const { return false; }
    virtual bool isTrackingRepaints() const { return false; }

#if HAVE(HDR_SUPPORT)
    virtual bool hdrForImagesEnabled() const { return false; }
#endif

    virtual bool shouldSkipLayerInDump(const GraphicsLayer*, OptionSet<LayerTreeAsTextOptions>) const { return false; }
    virtual bool shouldDumpPropertyForLayer(const GraphicsLayer*, ASCIILiteral, OptionSet<LayerTreeAsTextOptions>) const { return true; }

    virtual bool shouldAggressivelyRetainTiles(const GraphicsLayer*) const { return false; }
    virtual bool shouldTemporarilyRetainTileCohorts(const GraphicsLayer*) const { return true; }

    virtual bool useGiantTiles() const { return false; }
    virtual bool cssUnprefixedBackdropFilterEnabled() const { return false; }

    virtual bool needsPixelAligment() const { return false; }

    virtual bool needsIOSDumpRenderTreeMainFrameRenderViewLayerIsAlwaysOpaqueHack(const GraphicsLayer&) const { return false; }

    virtual void dumpProperties(const GraphicsLayer*, TextStream&, OptionSet<LayerTreeAsTextOptions>) const { }

    virtual void logFilledVisibleFreshTile(unsigned) { };

    virtual TransformationMatrix transformMatrixForProperty(AnimatedProperty) const { return { }; }

    virtual bool layerContainsBitmapOnly(const GraphicsLayer*) const { return false; }

    virtual bool layerNeedsPlatformContext(const GraphicsLayer*) const { return false; }

#ifndef NDEBUG
    // RenderLayerBacking overrides this to verify that it is not
    // currently painting contents. An ASSERT fails, if it is.
    // This is executed in GraphicsLayer construction and destruction
    // to verify that we don't create or destroy GraphicsLayers
    // while painting.
    virtual void verifyNotPainting() { }
#endif
};

} // namespace WebCore

