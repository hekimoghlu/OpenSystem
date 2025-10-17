/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

#if !PLATFORM(IOS_FAMILY)

#include "DrawingAreaProxy.h"
#include <wtf/RefCounted.h>

namespace WebKit {

class TiledCoreAnimationDrawingAreaProxy final : public DrawingAreaProxy, public RefCounted<TiledCoreAnimationDrawingAreaProxy> {
    WTF_MAKE_TZONE_ALLOCATED(TiledCoreAnimationDrawingAreaProxy);
    WTF_MAKE_NONCOPYABLE(TiledCoreAnimationDrawingAreaProxy);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TiledCoreAnimationDrawingAreaProxy);
public:
    static Ref<TiledCoreAnimationDrawingAreaProxy> create(WebPageProxy&, WebProcessProxy&);
    virtual ~TiledCoreAnimationDrawingAreaProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    TiledCoreAnimationDrawingAreaProxy(WebPageProxy&, WebProcessProxy&);

    // DrawingAreaProxy
    void deviceScaleFactorDidChange(CompletionHandler<void()>&&) override;
    void sizeDidChange() override;
    void colorSpaceDidChange() override;
    void minimumSizeForAutoLayoutDidChange() override;
    void sizeToContentAutoSizeMaximumSizeDidChange() override;

    void enterAcceleratedCompositingMode(uint64_t backingStoreStateID, const LayerTreeContext&) override;
    void updateAcceleratedCompositingMode(uint64_t backingStoreStateID, const LayerTreeContext&) override;
    void didFirstLayerFlush(uint64_t /* backingStoreStateID */, const LayerTreeContext&) override;

    void adjustTransientZoom(double scale, WebCore::FloatPoint origin) override;
    void commitTransientZoom(double scale, WebCore::FloatPoint origin) override;

    void waitForDidUpdateActivityState(ActivityStateChangeID) override;
    void dispatchPresentationCallbacksAfterFlushingLayers(IPC::Connection&, Vector<IPC::AsyncReplyID>&&) final;

    std::optional<WebCore::FramesPerSecond> displayNominalFramesPerSecond() final;

    void willSendUpdateGeometry();

    WTF::MachSendRight createFence() override;

    bool shouldSendWheelEventsToEventDispatcher() const override { return true; }

    void didUpdateGeometry();

    void sendUpdateGeometry();

    // Whether we're waiting for a DidUpdateGeometry message from the web process.
    bool m_isWaitingForDidUpdateGeometry { false };

    // The last size we sent to the web process.
    WebCore::IntSize m_lastSentSize;

    // The last minimum layout size we sent to the web process.
    WebCore::IntSize m_lastSentMinimumSizeForAutoLayout;

    // The last maxmium size for size-to-content auto-sizing we sent to the web process.
    WebCore::IntSize m_lastSentSizeToContentAutoSizeMaximumSize;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_DRAWING_AREA_PROXY(TiledCoreAnimationDrawingAreaProxy, DrawingAreaType::TiledCoreAnimation)

#endif // !PLATFORM(IOS_FAMILY)
