/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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

#include "PlatformCALayerRemote.h"
#include <WebCore/TileController.h>

namespace WebKit {

class PlatformCALayerRemoteTiledBacking final : public PlatformCALayerRemote {
    friend class PlatformCALayerRemote;
public:
    virtual ~PlatformCALayerRemoteTiledBacking();

private:
    PlatformCALayerRemoteTiledBacking(WebCore::PlatformCALayer::LayerType, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);

    WebCore::TiledBacking* tiledBacking() override { return m_tileController.get(); }

    void setNeedsDisplayInRect(const WebCore::FloatRect& dirtyRect) override;
    void setNeedsDisplay() override;

    const WebCore::PlatformCALayerList* customSublayers() const override;

    void setBounds(const WebCore::FloatRect&) override;
    
    bool isOpaque() const override;
    void setOpaque(bool) override;
    
    bool acceleratesDrawing() const override;
    void setAcceleratesDrawing(bool) override;

    WebCore::ContentsFormat contentsFormat() const override;
    void setContentsFormat(WebCore::ContentsFormat) override;

    float contentsScale() const override;
    void setContentsScale(float) override;
    
    void setBorderWidth(float) override;
    void setBorderColor(const WebCore::Color&) override;

    std::unique_ptr<WebCore::TileController> m_tileController;
    mutable WebCore::PlatformCALayerList m_customSublayers;
};

} // namespace WebKit
