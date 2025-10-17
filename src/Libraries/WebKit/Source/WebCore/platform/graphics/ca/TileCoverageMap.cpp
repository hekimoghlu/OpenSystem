/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
#include "TileCoverageMap.h"

#include "GraphicsContext.h"
#include "TileController.h"
#include "TileGrid.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TileCoverageMap);

TileCoverageMap::TileCoverageMap(const TileController& controller)
    : m_controller(controller)
    , m_updateTimer(*this, &TileCoverageMap::updateTimerFired)
    , m_layer(controller.rootLayer().createCompatibleLayer(PlatformCALayer::LayerType::LayerTypeSimpleLayer, this))
    , m_visibleViewportIndicatorLayer(controller.rootLayer().createCompatibleLayer(PlatformCALayer::LayerType::LayerTypeLayer, nullptr))
    , m_layoutViewportIndicatorLayer(controller.rootLayer().createCompatibleLayer(PlatformCALayer::LayerType::LayerTypeLayer, nullptr))
    , m_coverageRectIndicatorLayer(controller.rootLayer().createCompatibleLayer(PlatformCALayer::LayerType::LayerTypeLayer, nullptr))
    , m_position(FloatPoint(0, controller.topContentInset()))
{
    m_layer.get().setOpacity(0.75);
    m_layer.get().setAnchorPoint(FloatPoint3D());
    m_layer.get().setBorderColor(Color::black);
    m_layer.get().setBorderWidth(1);
    m_layer.get().setPosition(FloatPoint(2, 2));
    m_layer.get().setContentsScale(m_controller.deviceScaleFactor());

    m_visibleViewportIndicatorLayer.get().setName(MAKE_STATIC_STRING_IMPL("visible viewport indicator"));
    m_visibleViewportIndicatorLayer.get().setBorderWidth(2);
    m_visibleViewportIndicatorLayer.get().setAnchorPoint(FloatPoint3D());
    m_visibleViewportIndicatorLayer.get().setBorderColor(Color::red.colorWithAlphaByte(200));

    m_layoutViewportIndicatorLayer.get().setName(MAKE_STATIC_STRING_IMPL("layout viewport indicator"));
    m_layoutViewportIndicatorLayer.get().setBorderWidth(2);
    m_layoutViewportIndicatorLayer.get().setAnchorPoint(FloatPoint3D());
    m_layoutViewportIndicatorLayer.get().setBorderColor(SRGBA<uint8_t> { 0, 128, 128, 200 });
    
    m_coverageRectIndicatorLayer.get().setName(MAKE_STATIC_STRING_IMPL("coverage indicator"));
    m_coverageRectIndicatorLayer.get().setAnchorPoint(FloatPoint3D());
    m_coverageRectIndicatorLayer.get().setBackgroundColor(SRGBA<uint8_t> { 64, 64, 64, 50 });

    m_layer.get().appendSublayer(m_coverageRectIndicatorLayer);
    m_layer.get().appendSublayer(m_visibleViewportIndicatorLayer);
    
    if (m_controller.layoutViewportRect())
        m_layer.get().appendSublayer(m_layoutViewportIndicatorLayer);

    update();
}

TileCoverageMap::~TileCoverageMap()
{
    m_layer.get().setOwner(nullptr);
}

void TileCoverageMap::setNeedsUpdate()
{
    if (!m_updateTimer.isActive())
        m_updateTimer.startOneShot(0_s);
}

void TileCoverageMap::updateTimerFired()
{
    update();
}

void TileCoverageMap::update()
{
    FloatRect containerBounds = m_controller.bounds();
    FloatRect visibleRect = m_controller.visibleRect();
    FloatRect coverageRect = m_controller.coverageRect();
    visibleRect.contract(4, 4); // Layer is positioned 2px from top and left edges.

    float widthScale = 1;
    float scale = 1;
    if (!containerBounds.isEmpty()) {
        widthScale = std::min<float>(visibleRect.width() / containerBounds.width(), 0.1);
        float visibleHeight = visibleRect.height() - std::min(m_controller.topContentInset(), visibleRect.y());
        scale = std::min(widthScale, visibleHeight / containerBounds.height());
    }

    float indicatorScale = scale * m_controller.tileGrid().scale();

    FloatRect mapBounds = containerBounds;
    mapBounds.scale(indicatorScale);

    m_layer.get().setPosition(m_position + FloatPoint(2, 2));
    m_layer.get().setBounds(mapBounds);
    m_layer.get().setNeedsDisplay();

    visibleRect.scale(indicatorScale);
    visibleRect.expand(2, 2);
    m_visibleViewportIndicatorLayer->setPosition(visibleRect.location());
    m_visibleViewportIndicatorLayer->setBounds(FloatRect(FloatPoint(), visibleRect.size()));

    if (auto layoutViewportRect = m_controller.layoutViewportRect()) {
        FloatRect layoutRect = layoutViewportRect.value();
        layoutRect.scale(indicatorScale);
        layoutRect.expand(2, 2);
        m_layoutViewportIndicatorLayer->setPosition(layoutRect.location());
        m_layoutViewportIndicatorLayer->setBounds(FloatRect(FloatPoint(), layoutRect.size()));

        if (!m_layoutViewportIndicatorLayer->superlayer())
            m_layer.get().appendSublayer(m_layoutViewportIndicatorLayer);
    } else if (m_layoutViewportIndicatorLayer->superlayer())
        m_layoutViewportIndicatorLayer->removeFromSuperlayer();

    coverageRect.scale(indicatorScale);
    coverageRect.expand(2, 2);
    m_coverageRectIndicatorLayer->setPosition(coverageRect.location());
    m_coverageRectIndicatorLayer->setBounds(FloatRect(FloatPoint(), coverageRect.size()));

    Color visibleRectIndicatorColor;
    switch (m_controller.indicatorMode()) {
    case SynchronousScrollingBecauseOfLackOfScrollingCoordinatorIndication:
        visibleRectIndicatorColor = SRGBA<uint8_t> { 200, 80, 255 };
        break;
    case SynchronousScrollingBecauseOfStyleIndication:
        visibleRectIndicatorColor = Color::red;
        break;
    case SynchronousScrollingBecauseOfEventHandlersIndication:
        visibleRectIndicatorColor = Color::yellow;
        break;
    case AsyncScrollingIndication:
        visibleRectIndicatorColor = SRGBA<uint8_t> { 0, 200, 0 };
        break;
    }

    m_visibleViewportIndicatorLayer.get().setBorderColor(visibleRectIndicatorColor);
}

PlatformLayerIdentifier TileCoverageMap::platformCALayerIdentifier() const
{
    return m_layer->layerID();
}

void TileCoverageMap::platformCALayerPaintContents(PlatformCALayer* platformCALayer, GraphicsContext& context, const FloatRect&, OptionSet<GraphicsLayerPaintBehavior>)
{
    ASSERT_UNUSED(platformCALayer, platformCALayer == m_layer.ptr());
    m_controller.tileGrid().drawTileMapContents(context.platformContext(), m_layer.get().bounds());
}

float TileCoverageMap::platformCALayerDeviceScaleFactor() const
{
    return m_controller.rootLayer().owner()->platformCALayerDeviceScaleFactor();
}

void TileCoverageMap::setDeviceScaleFactor(float deviceScaleFactor)
{
    m_layer.get().setContentsScale(deviceScaleFactor);
}

}
