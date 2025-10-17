/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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

#include <WebCore/LayerHostingContextIdentifier.h>
#include <WebCore/ProcessIdentifier.h>
#include <wtf/HashMap.h>

namespace WebCore {
class BitmapTexture;
class TextureMapperPlatformLayer;
}

namespace WebKit {

class WCRemoteFrameHostLayerManager {
public:
    static WCRemoteFrameHostLayerManager& singleton();

    WebCore::TextureMapperPlatformLayer* acquireRemoteFrameHostLayer(WebCore::LayerHostingContextIdentifier, WebCore::ProcessIdentifier);
    void releaseRemoteFrameHostLayer(WebCore::LayerHostingContextIdentifier);

    void updateTexture(WebCore::LayerHostingContextIdentifier, WebCore::ProcessIdentifier, RefPtr<WebCore::BitmapTexture>);
    void removeAllLayersForProcess(WebCore::ProcessIdentifier);

private:
    class RemoteFrameHostLayerData;
    HashMap<WebCore::LayerHostingContextIdentifier, std::unique_ptr<RemoteFrameHostLayerData>> m_layers;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
