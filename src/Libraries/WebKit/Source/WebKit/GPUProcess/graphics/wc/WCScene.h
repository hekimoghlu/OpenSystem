/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

#include "UpdateInfo.h"
#include <WebCore/GraphicsLayer.h>
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/TextureMapperFPSCounter.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class TextureMapper;
class TextureMapperLayer;
class TextureMapperPlatformLayer;
class TextureMapperTiledBackingStore;
}

namespace WebKit {

class WCSceneContext;
struct WCUpdateInfo;

class WCScene {
    WTF_MAKE_TZONE_ALLOCATED(WCScene);
public:
    WCScene(WebCore::ProcessIdentifier, bool usesOffscreenRendering);
    ~WCScene();
    void initialize(WCSceneContext&);
    std::optional<UpdateInfo> update(WCUpdateInfo&&);

private:
    struct Layer;
    using LayerMap = HashMap<WebCore::PlatformLayerIdentifier, std::unique_ptr<Layer>>;

    WebCore::ProcessIdentifier m_webProcessIdentifier;
    WCSceneContext* m_context { nullptr };
    std::unique_ptr<WebCore::TextureMapper> m_textureMapper;
    WebCore::TextureMapperFPSCounter m_fpsCounter;
    LayerMap m_layers;
    bool m_usesOffscreenRendering;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
