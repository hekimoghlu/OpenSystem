/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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

#include "RemoteLayerTreeDrawingAreaProxy.h"

#if PLATFORM(IOS_FAMILY)

OBJC_CLASS WKDisplayLinkHandler;

namespace WebKit {

class RemoteLayerTreeDrawingAreaProxyIOS final : public RemoteLayerTreeDrawingAreaProxy {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerTreeDrawingAreaProxyIOS);
    WTF_MAKE_NONCOPYABLE(RemoteLayerTreeDrawingAreaProxyIOS);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteLayerTreeDrawingAreaProxyIOS);
public:
    static Ref<RemoteLayerTreeDrawingAreaProxyIOS> create(WebPageProxy&, WebProcessProxy&);
    virtual ~RemoteLayerTreeDrawingAreaProxyIOS();

    bool isRemoteLayerTreeDrawingAreaProxyIOS() const final { return true; }

    void scheduleDisplayRefreshCallbacksForAnimation();
    void pauseDisplayRefreshCallbacksForAnimation();

private:
    RemoteLayerTreeDrawingAreaProxyIOS(WebPageProxy&, WebProcessProxy&);

    WebCore::DelegatedScrollingMode delegatedScrollingMode() const override;

    std::unique_ptr<RemoteScrollingCoordinatorProxy> createScrollingCoordinatorProxy() const override;

    void setPreferredFramesPerSecond(IPC::Connection&, WebCore::FramesPerSecond) override;
    void scheduleDisplayRefreshCallbacks() override;
    void pauseDisplayRefreshCallbacks() override;

    void didRefreshDisplay() override;

    std::optional<WebCore::FramesPerSecond> displayNominalFramesPerSecond() override;

    WKDisplayLinkHandler *displayLinkHandler();

    RetainPtr<WKDisplayLinkHandler> m_displayLinkHandler;

    bool m_needsDisplayRefreshCallbacksForDrawing { false };
    bool m_needsDisplayRefreshCallbacksForAnimation { false };
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteLayerTreeDrawingAreaProxyIOS)
    static bool isType(const WebKit::DrawingAreaProxy& proxy) { return proxy.isRemoteLayerTreeDrawingAreaProxyIOS(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // PLATFORM(IOS_FAMILY)
