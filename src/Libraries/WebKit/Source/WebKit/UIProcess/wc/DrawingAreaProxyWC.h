/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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

#include "BackingStore.h"
#include "DrawingAreaProxy.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
class Region;
}

namespace WebKit {

class DrawingAreaProxyWC final : public DrawingAreaProxy, public RefCounted<DrawingAreaProxyWC> {
    WTF_MAKE_TZONE_ALLOCATED(DrawingAreaProxyWC);
    WTF_MAKE_NONCOPYABLE(DrawingAreaProxyWC);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DrawingAreaProxyWC);
public:
    static Ref<DrawingAreaProxyWC> create(WebPageProxy&, WebProcessProxy&);

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void paint(PlatformPaintContextPtr, const WebCore::IntRect&, WebCore::Region& unpaintedRegion);

private:
    DrawingAreaProxyWC(WebPageProxy&, WebProcessProxy&);

    // DrawingAreaProxy
    void deviceScaleFactorDidChange(CompletionHandler<void()>&&) override;
    void sizeDidChange() override;
    bool shouldSendWheelEventsToEventDispatcher() const final { return true; }

    // message handers
    void update(uint64_t, UpdateInfo&&) override;
    void enterAcceleratedCompositingMode(uint64_t, const LayerTreeContext&) override;

    void incorporateUpdate(UpdateInfo&&);
    void discardBackingStore();

    uint64_t m_currentBackingStoreStateID { 0 };
    std::optional<BackingStore> m_backingStore;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_DRAWING_AREA_PROXY(DrawingAreaProxyWC, DrawingAreaType::WC)

#endif // USE(GRAPHICS_LAYER_WC)
