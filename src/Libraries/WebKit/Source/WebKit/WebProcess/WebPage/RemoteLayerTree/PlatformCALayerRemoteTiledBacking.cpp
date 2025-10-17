/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#import "PlatformCALayerRemoteTiledBacking.h"

#import "RemoteLayerTreeContext.h"
#import <WebCore/GraphicsLayerCA.h>
#import <WebCore/PlatformCALayerCocoa.h>
#import <wtf/RetainPtr.h>

namespace WebKit {
using namespace WebCore;

PlatformCALayerRemoteTiledBacking::PlatformCALayerRemoteTiledBacking(LayerType layerType, PlatformCALayerClient* owner, RemoteLayerTreeContext& context)
    : PlatformCALayerRemote(layerType, owner, context)
    , m_tileController(makeUnique<TileController>(this, WebCore::TileController::AllowScrollPerformanceLogging::No))
{
    PlatformCALayerRemote::setContentsScale(m_tileController->contentsScale());
}

PlatformCALayerRemoteTiledBacking::~PlatformCALayerRemoteTiledBacking()
{
}

void PlatformCALayerRemoteTiledBacking::setNeedsDisplayInRect(const FloatRect& dirtyRect)
{
    m_tileController->setNeedsDisplayInRect(enclosingIntRect(dirtyRect));
}

void PlatformCALayerRemoteTiledBacking::setNeedsDisplay()
{
    m_tileController->setNeedsDisplay();
}

const WebCore::PlatformCALayerList* PlatformCALayerRemoteTiledBacking::customSublayers() const
{
    m_customSublayers = m_tileController->containerLayers();
    return &m_customSublayers;
}

void PlatformCALayerRemoteTiledBacking::setBounds(const WebCore::FloatRect& bounds)
{
    PlatformCALayerRemote::setBounds(bounds);
    m_tileController->tileCacheLayerBoundsChanged();
}

bool PlatformCALayerRemoteTiledBacking::isOpaque() const
{
    return m_tileController->tilesAreOpaque();
}

void PlatformCALayerRemoteTiledBacking::setOpaque(bool opaque)
{
    m_tileController->setTilesOpaque(opaque);
}

bool PlatformCALayerRemoteTiledBacking::acceleratesDrawing() const
{
    return m_tileController->acceleratesDrawing();
}

void PlatformCALayerRemoteTiledBacking::setAcceleratesDrawing(bool acceleratesDrawing)
{
    m_tileController->setAcceleratesDrawing(acceleratesDrawing);
}

ContentsFormat PlatformCALayerRemoteTiledBacking::contentsFormat() const
{
    return m_tileController->contentsFormat();
}

void PlatformCALayerRemoteTiledBacking::setContentsFormat(ContentsFormat contentsFormat)
{
    m_tileController->setContentsFormat(contentsFormat);
}

float PlatformCALayerRemoteTiledBacking::contentsScale() const
{
    return m_tileController->contentsScale();
}

void PlatformCALayerRemoteTiledBacking::setContentsScale(float scale)
{
    PlatformCALayerRemote::setContentsScale(scale);
    m_tileController->setContentsScale(scale);
}

void PlatformCALayerRemoteTiledBacking::setBorderWidth(float borderWidth)
{
    m_tileController->setTileDebugBorderWidth(borderWidth / 2);
}

void PlatformCALayerRemoteTiledBacking::setBorderColor(const WebCore::Color& color)
{
    m_tileController->setTileDebugBorderColor(color);
}

} // namespace WebKit
