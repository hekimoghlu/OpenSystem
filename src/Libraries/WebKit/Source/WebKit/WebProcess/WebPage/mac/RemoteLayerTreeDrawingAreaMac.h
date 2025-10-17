/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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

#include "RemoteLayerTreeDrawingArea.h"
#include <wtf/TZoneMalloc.h>

#if PLATFORM(MAC)

namespace WebCore {
class TiledBacking;
}

namespace WebKit {

class RemoteLayerTreeDrawingAreaMac final : public RemoteLayerTreeDrawingArea {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerTreeDrawingAreaMac);
public:
    static Ref<RemoteLayerTreeDrawingAreaMac> create(WebPage& webPage, const WebPageCreationParameters& parameters)
    {
        return adoptRef(*new RemoteLayerTreeDrawingAreaMac(webPage, parameters));
    }

    virtual ~RemoteLayerTreeDrawingAreaMac();

private:
    RemoteLayerTreeDrawingAreaMac(WebPage&, const WebPageCreationParameters&);

    WebCore::DelegatedScrollingMode delegatedScrollingMode() const final;

    void setColorSpace(std::optional<WebCore::DestinationColorSpace>) final;
    std::optional<WebCore::DestinationColorSpace> displayColorSpace() const final;

    std::optional<WebCore::DestinationColorSpace> m_displayColorSpace;

    bool usesDelegatedPageScaling() const override { return false; }

    void mainFrameContentSizeChanged(WebCore::FrameIdentifier, const WebCore::IntSize&) final;

    void adjustTransientZoom(double scale, WebCore::FloatPoint origin) final;

    void willCommitLayerTree(RemoteLayerTreeTransaction&) final;
};

} // namespace WebKit

#endif // PLATFORM(MAC)
