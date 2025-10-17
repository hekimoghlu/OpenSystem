/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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

#if USE(GRAPHICS_LAYER_WC)

#include "WCBackingStore.h"
#include "WCContentBufferIdentifier.h"
#include <WebCore/GraphicsLayer.h>
#include <WebCore/TextureMapperSparseBackingStore.h>
#include <optional>
#include <wtf/OptionSet.h>

namespace WebKit {

struct WCTileUpdate {
    WebCore::TextureMapperSparseBackingStore::TileIndex index;
    bool willRemove { false };
    WCBackingStore backingStore;
    WebCore::IntRect dirtyRect;
};

enum class WCLayerChange : uint32_t {
    Children                     = 1 <<  0,
    MaskLayer                    = 1 <<  1,
    ReplicaLayer                 = 1 <<  2,
    Position                     = 1 <<  3,
    AnchorPoint                  = 1 <<  4,
    Size                         = 1 <<  5,
    BoundsOrigin                 = 1 <<  6,
    MasksToBounds                = 1 <<  7,
    ContentsRectClipsDescendants = 1 <<  8,
    ShowDebugBorder              = 1 <<  9,
    ShowRepaintCounter           = 1 << 10,
    ContentsVisible              = 1 << 11,
    BackfaceVisibility           = 1 << 12,
    Preserves3D                  = 1 << 13,
    SolidColor                   = 1 << 14,
    DebugBorderColor             = 1 << 15,
    Opacity                      = 1 << 16,
    DebugBorderWidth             = 1 << 17,
    RepaintCount                 = 1 << 18,
    ContentsRect                 = 1 << 19,
    Background                   = 1 << 20,
    Transform                    = 1 << 21,
    ChildrenTransform            = 1 << 22,
    Filters                      = 1 << 23,
    BackdropFilters              = 1 << 24,
    BackdropFiltersRect          = 1 << 25,
    ContentsClippingRect         = 1 << 26,
    PlatformLayer                = 1 << 27,
    RemoteFrame                  = 1 << 28,
};

struct WCLayerUpdateInfo {
    WebCore::PlatformLayerIdentifier id;
    OptionSet<WCLayerChange> changes { };
    Vector<WebCore::PlatformLayerIdentifier> children { };
    std::optional<WebCore::PlatformLayerIdentifier> maskLayer { };
    std::optional<WebCore::PlatformLayerIdentifier> replicaLayer { };
    WebCore::FloatPoint position { };
    WebCore::FloatPoint3D anchorPoint { };
    WebCore::FloatSize size { };
    WebCore::FloatPoint boundsOrigin { };
    bool masksToBounds { false };
    bool contentsRectClipsDescendants { false };
    bool showDebugBorder { false };
    bool showRepaintCounter { false };
    bool contentsVisible { false };
    bool backfaceVisibility { false };
    bool preserves3D { false };
    WebCore::Color solidColor { };
    WebCore::Color debugBorderColor { };
    float opacity { 0 };
    float debugBorderWidth { 0 };
    int repaintCount { 0 };
    WebCore::FloatRect contentsRect { };

    struct BackgroundChanges {
        WebCore::Color color;
        bool hasBackingStore;
        WebCore::IntSize backingStoreSize;
        Vector<WCTileUpdate> tileUpdates;
    } background { };

    WebCore::TransformationMatrix transform { };
    WebCore::TransformationMatrix childrenTransform { };
    WebCore::FilterOperations filters { };
    WebCore::FilterOperations backdropFilters { };
    WebCore::FloatRoundedRect backdropFiltersRect { };
    WebCore::FloatRoundedRect contentsClippingRect { };

    struct PlatformLayerChanges {
        bool hasLayer;
        Vector<WCContentBufferIdentifier> identifiers;
    } platformLayer { };

    Markable<WebCore::LayerHostingContextIdentifier> hostIdentifier { };
};

struct WCUpdateInfo {
    WebCore::IntSize viewport;
    Markable<WebCore::LayerHostingContextIdentifier> remoteContextHostedIdentifier;
    Markable<WebCore::PlatformLayerIdentifier> rootLayer;
    Vector<WebCore::PlatformLayerIdentifier> addedLayers;
    Vector<WebCore::PlatformLayerIdentifier> removedLayers;
    Vector<WCLayerUpdateInfo> changedLayers;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
