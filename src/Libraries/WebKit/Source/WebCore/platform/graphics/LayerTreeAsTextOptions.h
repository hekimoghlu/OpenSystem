/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#include <wtf/OptionSet.h>

namespace WebCore {

enum class LayerTreeAsTextOptions : uint16_t {
    Debug                        = 1 << 0, // Dump extra debugging info like layer addresses.
    IncludeVisibleRects          = 1 << 1,
    IncludeTileCaches            = 1 << 2,
    IncludeRepaintRects          = 1 << 3,
    IncludePaintingPhases        = 1 << 4,
    IncludeContentLayers         = 1 << 5,
    IncludePageOverlayLayers     = 1 << 6,
    IncludeAcceleratesDrawing    = 1 << 7,
    IncludeClipping              = 1 << 8,
    IncludeBackingStoreAttached  = 1 << 9,
    IncludeRootLayerProperties   = 1 << 10,
    IncludeEventRegion           = 1 << 11,
    IncludeExtendedColor         = 1 << 12,
    IncludeDeviceScale           = 1 << 13,
    IncludeRootLayers            = 1 << 14,
};

static constexpr OptionSet<LayerTreeAsTextOptions> AllLayerTreeAsTextOptions = {
    LayerTreeAsTextOptions::Debug,
    LayerTreeAsTextOptions::IncludeVisibleRects,
    LayerTreeAsTextOptions::IncludeTileCaches,
    LayerTreeAsTextOptions::IncludeRepaintRects,
    LayerTreeAsTextOptions::IncludePaintingPhases,
    LayerTreeAsTextOptions::IncludeContentLayers,
    LayerTreeAsTextOptions::IncludePageOverlayLayers,
    LayerTreeAsTextOptions::IncludeAcceleratesDrawing,
    LayerTreeAsTextOptions::IncludeClipping,
    LayerTreeAsTextOptions::IncludeBackingStoreAttached,
    LayerTreeAsTextOptions::IncludeRootLayerProperties,
    LayerTreeAsTextOptions::IncludeEventRegion,
    LayerTreeAsTextOptions::IncludeExtendedColor,
};

} // namespace WebCore
