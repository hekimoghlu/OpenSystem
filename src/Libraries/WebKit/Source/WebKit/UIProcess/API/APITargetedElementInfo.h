/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

#include "APIObject.h"
#include <WebCore/ElementTargetingTypes.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class ShareableBitmapHandle;
}

namespace WebKit {
class WebPageProxy;
}

namespace API {

class FrameTreeNode;

class TargetedElementInfo final : public ObjectImpl<Object::Type::TargetedElementInfo> {
public:
    static Ref<TargetedElementInfo> create(WebKit::WebPageProxy& page, WebCore::TargetedElementInfo&& info)
    {
        return adoptRef(*new TargetedElementInfo(page, WTFMove(info)));
    }

    explicit TargetedElementInfo(WebKit::WebPageProxy&, WebCore::TargetedElementInfo&&);

    WebCore::RectEdges<bool> offsetEdges() const { return m_info.offsetEdges; }

    const WTF::String& renderedText() const { return m_info.renderedText; }
    const WTF::String& searchableText() const { return m_info.searchableText; }
    const WTF::String& screenReaderText() const { return m_info.screenReaderText; }
    const Vector<Vector<WTF::String>>& selectors() const { return m_info.selectors; }
    WebCore::PositionType positionType() const { return m_info.positionType; }
    WebCore::FloatRect boundsInRootView() const { return m_info.boundsInRootView; }
    WebCore::FloatRect boundsInWebView() const;
    WebCore::FloatRect boundsInClientCoordinates() const { return m_info.boundsInClientCoordinates; }

    bool isNearbyTarget() const { return m_info.isNearbyTarget; }
    bool isPseudoElement() const { return m_info.isPseudoElement; }
    bool isInShadowTree() const { return m_info.isInShadowTree; }
    bool isInVisibilityAdjustmentSubtree() const { return m_info.isInVisibilityAdjustmentSubtree; }
    bool hasLargeReplacedDescendant() const { return m_info.hasLargeReplacedDescendant; }
    bool hasAudibleMedia() const { return m_info.hasAudibleMedia; }

    const HashSet<WTF::URL>& mediaAndLinkURLs() const { return m_info.mediaAndLinkURLs; }

    void childFrames(CompletionHandler<void(Vector<Ref<FrameTreeNode>>&&)>&&) const;

    bool isSameElement(const TargetedElementInfo&) const;

    WebCore::ElementIdentifier elementIdentifier() const { return m_info.elementIdentifier; }
    WebCore::ScriptExecutionContextIdentifier documentIdentifier() const { return m_info.documentIdentifier; }

    void takeSnapshot(CompletionHandler<void(std::optional<WebCore::ShareableBitmapHandle>&&)>&&);

private:
    WebCore::TargetedElementInfo m_info;
    WeakPtr<WebKit::WebPageProxy> m_page;
};

} // namespace API
