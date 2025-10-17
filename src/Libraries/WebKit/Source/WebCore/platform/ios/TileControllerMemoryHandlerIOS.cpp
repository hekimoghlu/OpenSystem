/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
#include "TileControllerMemoryHandlerIOS.h"

#if PLATFORM(IOS_FAMILY)

#include "TileController.h"
#include <wtf/MemoryPressureHandler.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static const unsigned kMaxCountOfUnparentedTiledLayers = 16;

TileControllerMemoryHandler::TileControllerMemoryHandler() = default;

TileControllerMemoryHandler::~TileControllerMemoryHandler() = default;

void TileControllerMemoryHandler::removeTileController(TileController* controller)
{
    if (m_tileControllers.contains(controller))
        m_tileControllers.remove(controller);
}

unsigned TileControllerMemoryHandler::totalUnparentedTiledLayers() const
{
    unsigned totalUnparentedLayers = 0;
    for (auto& tileController : m_tileControllers)
        totalUnparentedLayers += tileController->numberOfUnparentedTiles();
    return totalUnparentedLayers;
}

void TileControllerMemoryHandler::tileControllerGainedUnparentedTiles(TileController* controller)
{
    m_tileControllers.appendOrMoveToLast(controller);

    // If we are under memory pressure, remove all unparented tiles now.
    if (MemoryPressureHandler::singleton().isUnderMemoryPressure()) {
        trimUnparentedTilesToTarget(0);
        return;
    }

    if (totalUnparentedTiledLayers() < kMaxCountOfUnparentedTiledLayers)
        return;

    trimUnparentedTilesToTarget(kMaxCountOfUnparentedTiledLayers);
}

void TileControllerMemoryHandler::trimUnparentedTilesToTarget(int target)
{
    while (!m_tileControllers.isEmpty()) {
        m_tileControllers.first()->removeUnparentedTilesNow();
        m_tileControllers.removeFirst();

        if (target > 0 && totalUnparentedTiledLayers() < static_cast<unsigned>(target))
            return;
    }
}

TileControllerMemoryHandler& tileControllerMemoryHandler()
{
    static NeverDestroyed<TileControllerMemoryHandler> staticTileControllerMemoryHandler;
    return staticTileControllerMemoryHandler;
}

}

#endif // PLATFORM(IOS_FAMILY)
