/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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

#if PLATFORM(IOS_FAMILY)

#include <wtf/CheckedPtr.h>
#include <wtf/ListHashSet.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class TileController;

class TileControllerMemoryHandler {
public:
    friend TileControllerMemoryHandler& tileControllerMemoryHandler();
    friend class NeverDestroyed<TileControllerMemoryHandler>;

    ~TileControllerMemoryHandler();

    void removeTileController(TileController*);
    void tileControllerGainedUnparentedTiles(TileController*);
    WEBCORE_EXPORT void trimUnparentedTilesToTarget(int target);

private:
    TileControllerMemoryHandler();

    unsigned totalUnparentedTiledLayers() const;

    using TileControllerList = ListHashSet<CheckedPtr<TileController>>;
    TileControllerList m_tileControllers;
};

WEBCORE_EXPORT TileControllerMemoryHandler& tileControllerMemoryHandler();
}

#endif // PLATFORM(IOS_FAMILY)
