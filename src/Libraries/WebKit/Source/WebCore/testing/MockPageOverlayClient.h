/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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

#include "MockPageOverlay.h"
#include "PageOverlay.h"
#include <wtf/HashSet.h>

namespace WebCore {

class LocalFrame;
class Page;
enum class LayerTreeAsTextOptions : uint16_t;

class MockPageOverlayClient final : public PageOverlayClient {
    friend class NeverDestroyed<MockPageOverlayClient>;
public:
    static MockPageOverlayClient& singleton();

    explicit MockPageOverlayClient();

    Ref<MockPageOverlay> installOverlay(Page&, PageOverlay::OverlayType);
    void uninstallAllOverlays();

    String layerTreeAsText(Page&, OptionSet<LayerTreeAsTextOptions>);

    virtual ~MockPageOverlayClient() = default;

private:
    void willMoveToPage(PageOverlay&, Page*) override;
    void didMoveToPage(PageOverlay&, Page*) override;
    void drawRect(PageOverlay&, GraphicsContext&, const IntRect& dirtyRect) override;
    bool mouseEvent(PageOverlay&, const PlatformMouseEvent&) override;
    void didScrollFrame(PageOverlay&, LocalFrame&) override;

    bool copyAccessibilityAttributeStringValueForPoint(PageOverlay&, String /* attribute */, FloatPoint, String&) override;
    bool copyAccessibilityAttributeBoolValueForPoint(PageOverlay&, String /* attribute */, FloatPoint, bool&) override;
    Vector<String> copyAccessibilityAttributeNames(PageOverlay&, bool /* parameterizedNames */) override;

    HashSet<RefPtr<MockPageOverlay>> m_overlays;
};

} // namespace WebCore
