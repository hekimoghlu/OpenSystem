/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#if ENABLE(UI_SIDE_COMPOSITING)

#include "TransactionID.h"
#include <WebCore/FloatRect.h>
#include <WebCore/LengthBox.h>
#include <WebCore/VelocityData.h>
#include <wtf/MonotonicTime.h>
#include <wtf/OptionSet.h>
#include <wtf/text/WTFString.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WTF {
class TextStream;
}

namespace WebKit {

enum class ViewStabilityFlag : uint8_t {
    ScrollViewInteracting               = 1 << 0, // Dragging, zooming, interrupting deceleration
    ScrollViewAnimatedScrollOrZoom      = 1 << 1, // Decelerating, scrolling to top, animated zoom
    ScrollViewRubberBanding             = 1 << 2,
    ChangingObscuredInsetsInteractively = 1 << 3,
    UnstableForTesting                  = 1 << 4
};

class VisibleContentRectUpdateInfo {
public:
    VisibleContentRectUpdateInfo() = default;

    VisibleContentRectUpdateInfo(const WebCore::FloatRect& exposedContentRect, const WebCore::FloatRect& unobscuredContentRect, const WebCore::FloatBoxExtent& contentInsets,
        const WebCore::FloatRect& unobscuredRectInScrollViewCoordinates, const WebCore::FloatRect& unobscuredContentRectRespectingInputViewBounds, const WebCore::FloatRect& layoutViewportRect,
        const WebCore::FloatBoxExtent& obscuredInsets, const WebCore::FloatBoxExtent& unobscuredSafeAreaInsets, double scale, OptionSet<ViewStabilityFlag> viewStability,
        bool isFirstUpdateForNewViewSize, bool allowShrinkToFit, bool enclosedInScrollableAncestorView, const WebCore::VelocityData& scrollVelocity, TransactionID lastLayerTreeTransactionId)
        : m_exposedContentRect(exposedContentRect)
        , m_unobscuredContentRect(unobscuredContentRect)
        , m_contentInsets(contentInsets)
        , m_unobscuredContentRectRespectingInputViewBounds(unobscuredContentRectRespectingInputViewBounds)
        , m_unobscuredRectInScrollViewCoordinates(unobscuredRectInScrollViewCoordinates)
        , m_layoutViewportRect(layoutViewportRect)
        , m_obscuredInsets(obscuredInsets)
        , m_unobscuredSafeAreaInsets(unobscuredSafeAreaInsets)
        , m_scrollVelocity(scrollVelocity)
        , m_lastLayerTreeTransactionID(lastLayerTreeTransactionId)
        , m_scale(scale)
        , m_viewStability(viewStability)
        , m_isFirstUpdateForNewViewSize(isFirstUpdateForNewViewSize)
        , m_allowShrinkToFit(allowShrinkToFit)
        , m_enclosedInScrollableAncestorView(enclosedInScrollableAncestorView)
    {
    }

    const WebCore::FloatRect& exposedContentRect() const { return m_exposedContentRect; }
    const WebCore::FloatRect& unobscuredContentRect() const { return m_unobscuredContentRect; }
    const WebCore::FloatBoxExtent& contentInsets() const { return m_contentInsets; }
    const WebCore::FloatRect& unobscuredRectInScrollViewCoordinates() const { return m_unobscuredRectInScrollViewCoordinates; }
    const WebCore::FloatRect& unobscuredContentRectRespectingInputViewBounds() const { return m_unobscuredContentRectRespectingInputViewBounds; }
    const WebCore::FloatRect& layoutViewportRect() const { return m_layoutViewportRect; }
    const WebCore::FloatBoxExtent& obscuredInsets() const { return m_obscuredInsets; }
    const WebCore::FloatBoxExtent& unobscuredSafeAreaInsets() const { return m_unobscuredSafeAreaInsets; }

    double scale() const { return m_scale; }
    bool inStableState() const { return m_viewStability.isEmpty(); }
    OptionSet<ViewStabilityFlag> viewStability() const { return m_viewStability; }
    bool isFirstUpdateForNewViewSize() const { return m_isFirstUpdateForNewViewSize; }
    bool allowShrinkToFit() const { return m_allowShrinkToFit; }
    bool enclosedInScrollableAncestorView() const { return m_enclosedInScrollableAncestorView; }
    const WebCore::VelocityData& scrollVelocity() const { return m_scrollVelocity; }
    TransactionID lastLayerTreeTransactionID() const { return m_lastLayerTreeTransactionID; }

    MonotonicTime timestamp() const { return m_scrollVelocity.lastUpdateTime; }

    String dump() const;

private:
    WebCore::FloatRect m_exposedContentRect;
    WebCore::FloatRect m_unobscuredContentRect;
    WebCore::FloatBoxExtent m_contentInsets;
    WebCore::FloatRect m_unobscuredContentRectRespectingInputViewBounds;
    WebCore::FloatRect m_unobscuredRectInScrollViewCoordinates;
    WebCore::FloatRect m_layoutViewportRect;
    WebCore::FloatBoxExtent m_obscuredInsets;
    WebCore::FloatBoxExtent m_unobscuredSafeAreaInsets;
    WebCore::VelocityData m_scrollVelocity;
    TransactionID m_lastLayerTreeTransactionID;
    double m_scale { -1 };
    OptionSet<ViewStabilityFlag> m_viewStability;
    bool m_isFirstUpdateForNewViewSize { false };
    bool m_allowShrinkToFit { false };
    bool m_enclosedInScrollableAncestorView { false };
};

inline bool operator==(const VisibleContentRectUpdateInfo& a, const VisibleContentRectUpdateInfo& b)
{
    // Note: the comparison doesn't include timestamp and velocity since we care about equality based on the other data.
    return a.scale() == b.scale()
        && a.exposedContentRect() == b.exposedContentRect()
        && a.unobscuredContentRect() == b.unobscuredContentRect()
        && a.contentInsets() == b.contentInsets()
        && a.unobscuredContentRectRespectingInputViewBounds() == b.unobscuredContentRectRespectingInputViewBounds()
        && a.layoutViewportRect() == b.layoutViewportRect()
        && a.obscuredInsets() == b.obscuredInsets()
        && a.unobscuredSafeAreaInsets() == b.unobscuredSafeAreaInsets()
        && a.scrollVelocity().equalIgnoringTimestamp(b.scrollVelocity())
        && a.viewStability() == b.viewStability()
        && a.isFirstUpdateForNewViewSize() == b.isFirstUpdateForNewViewSize()
        && a.allowShrinkToFit() == b.allowShrinkToFit()
        && a.enclosedInScrollableAncestorView() == b.enclosedInScrollableAncestorView();
}

WTF::TextStream& operator<<(WTF::TextStream&, ViewStabilityFlag);
WTF::TextStream& operator<<(WTF::TextStream&, const VisibleContentRectUpdateInfo&);

} // namespace WebKit

#endif // ENABLE(UI_SIDE_COMPOSITING)
