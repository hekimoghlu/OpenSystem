/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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

#if PLATFORM(MAC)

#include "DisplayLinkObserverID.h"
#include <WebCore/AnimationFrameRate.h>

namespace WebKit {

class DisplayLink;
class RemoteLayerTreeDisplayLinkClient;
class RemoteLayerTreeTransaction;
class RemoteScrollingCoordinatorProxy;
class RemoteScrollingCoordinatorTransaction;

class RemoteLayerTreeDrawingAreaProxyMac final : public RemoteLayerTreeDrawingAreaProxy {
friend class RemoteScrollingCoordinatorProxyMac;
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerTreeDrawingAreaProxyMac);
    WTF_MAKE_NONCOPYABLE(RemoteLayerTreeDrawingAreaProxyMac);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteLayerTreeDrawingAreaProxyMac);
public:
    static Ref<RemoteLayerTreeDrawingAreaProxyMac> create(WebPageProxy&, WebProcessProxy&);
    ~RemoteLayerTreeDrawingAreaProxyMac();

    void didRefreshDisplay() override;

    DisplayLink& displayLink();
    DisplayLink* existingDisplayLink();

    void updateZoomTransactionID();
    std::optional<WebCore::PlatformLayerIdentifier> pageScalingLayerID() { return m_pageScalingLayerID.asOptional(); }
    std::optional<WebCore::PlatformLayerIdentifier> pageScrollingLayerID() { return m_pageScrollingLayerID.asOptional(); }

private:
    RemoteLayerTreeDrawingAreaProxyMac(WebPageProxy&, WebProcessProxy&);

    WebCore::DelegatedScrollingMode delegatedScrollingMode() const override;
    std::unique_ptr<RemoteScrollingCoordinatorProxy> createScrollingCoordinatorProxy() const override;

    bool isRemoteLayerTreeDrawingAreaProxyMac() const override { return true; }

    void layoutBannerLayers(const RemoteLayerTreeTransaction&);

    void didCommitLayerTree(IPC::Connection&, const RemoteLayerTreeTransaction&, const RemoteScrollingCoordinatorTransaction&) override;

    void adjustTransientZoom(double, WebCore::FloatPoint) override;
    void commitTransientZoom(double, WebCore::FloatPoint) override;

    void sendCommitTransientZoom(double, WebCore::FloatPoint, std::optional<WebCore::ScrollingNodeID>);

    void applyTransientZoomToLayer();
    void removeTransientZoomFromLayer();

    void scheduleDisplayRefreshCallbacks() override;
    void pauseDisplayRefreshCallbacks() override;
    void setPreferredFramesPerSecond(IPC::Connection&, WebCore::FramesPerSecond) override;
    void windowScreenDidChange(WebCore::PlatformDisplayID) override;
    std::optional<WebCore::FramesPerSecond> displayNominalFramesPerSecond() override;

    void dispatchSetTopContentInset() override;

    void colorSpaceDidChange() override;

    void viewIsBecomingVisible() final;
    void viewIsBecomingInvisible() final;

    void didChangeViewExposedRect() override;

    void setDisplayLinkWantsFullSpeedUpdates(bool) override;

    void removeObserver(std::optional<DisplayLinkObserverID>&);

    WTF::MachSendRight createFence() override;

    std::optional<WebCore::PlatformDisplayID> m_displayID; // Would be nice to make this non-optional, and ensure we always get one on creation.
    std::optional<WebCore::FramesPerSecond> m_displayNominalFramesPerSecond;
    WebCore::FramesPerSecond m_clientPreferredFramesPerSecond { WebCore::FullSpeedFramesPerSecond };

    std::optional<DisplayLinkObserverID> m_displayRefreshObserverID;
    std::optional<DisplayLinkObserverID> m_fullSpeedUpdateObserverID;
    std::unique_ptr<RemoteLayerTreeDisplayLinkClient> m_displayLinkClient;

    Markable<WebCore::PlatformLayerIdentifier> m_pageScalingLayerID;
    Markable<WebCore::PlatformLayerIdentifier> m_pageScrollingLayerID;

    bool m_usesOverlayScrollbars { false };
    bool m_shouldLogNextObserverChange { false };
    bool m_shouldLogNextDisplayRefresh { false };

    std::optional<WebCore::ScrollbarStyle> m_scrollbarStyle;

    std::optional<TransactionID> m_transactionIDAfterEndingTransientZoom;
    std::optional<double> m_transientZoomScale;
    std::optional<WebCore::FloatPoint> m_transientZoomOrigin;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteLayerTreeDrawingAreaProxyMac)
    static bool isType(const WebKit::DrawingAreaProxy& proxy) { return proxy.isRemoteLayerTreeDrawingAreaProxyMac(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // #if PLATFORM(MAC)
