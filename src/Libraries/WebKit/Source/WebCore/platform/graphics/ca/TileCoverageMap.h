/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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
#ifndef TileCoverageMap_h
#define TileCoverageMap_h

#include "FloatRect.h"
#include "IntRect.h"
#include "PlatformCALayer.h"
#include "PlatformCALayerClient.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FloatRect;
class IntPoint;
class IntRect;
class TileController;

class TileCoverageMap final : public PlatformCALayerClient, public CanMakeCheckedPtr<TileCoverageMap> {
    WTF_MAKE_TZONE_ALLOCATED(TileCoverageMap);
    WTF_MAKE_NONCOPYABLE(TileCoverageMap);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TileCoverageMap);
public:
    TileCoverageMap(const TileController&);
    ~TileCoverageMap();

    void update();
    void setPosition(const FloatPoint& position) { m_position = position; }

    PlatformCALayer& layer() { return m_layer; }

    void setDeviceScaleFactor(float);

    void setNeedsUpdate();

private:
    // PlatformCALayerClient
    PlatformLayerIdentifier platformCALayerIdentifier() const override;
    GraphicsLayer::CompositingCoordinatesOrientation platformCALayerContentsOrientation() const override { return GraphicsLayer::CompositingCoordinatesOrientation::TopDown; }
    bool platformCALayerContentsOpaque() const override { return true; }
    bool platformCALayerDrawsContent() const override { return true; }
    void platformCALayerPaintContents(PlatformCALayer*, GraphicsContext&, const FloatRect&, OptionSet<GraphicsLayerPaintBehavior>) override;
    float platformCALayerDeviceScaleFactor() const override;

    void updateTimerFired();
    
    const TileController& m_controller;
    
    Timer m_updateTimer;

    Ref<PlatformCALayer> m_layer;
    Ref<PlatformCALayer> m_visibleViewportIndicatorLayer;
    Ref<PlatformCALayer> m_layoutViewportIndicatorLayer;
    Ref<PlatformCALayer> m_coverageRectIndicatorLayer;

    FloatPoint m_position;
};

}

#endif
