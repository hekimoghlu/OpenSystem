/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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

#if PLATFORM(IOS_FAMILY)

#include "ArgumentCoders.h"
#include "CursorContext.h"
#include "InteractionInformationRequest.h"
#include <WebCore/ElementAnimationContext.h>
#include <WebCore/ElementContext.h>
#include <WebCore/IntPoint.h>
#include <WebCore/ScrollTypes.h>
#include <WebCore/SelectionGeometry.h>
#include <WebCore/ShareableBitmap.h>
#include <WebCore/TextIndicator.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

struct InteractionInformationAtPosition {
    static InteractionInformationAtPosition invalidInformation()
    {
        InteractionInformationAtPosition response;
        response.canBeValid = false;
        return response;
    }

    InteractionInformationRequest request;

    bool canBeValid { true };
    std::optional<bool> nodeAtPositionHasDoubleClickHandler;

    enum class Selectability : uint8_t {
        Selectable,
        UnselectableDueToFocusableElement,
        UnselectableDueToLargeElementBounds,
        UnselectableDueToUserSelectNone,
        UnselectableDueToMediaControls,
    };

    Selectability selectability { Selectability::Selectable };

    bool isSelected { false };
    bool prefersDraggingOverTextSelection { false };
    bool isNearMarkedText { false };
    bool touchCalloutEnabled { true };
    bool isLink { false };
    bool isImage { false };
    bool isAttachment { false };
    bool isAnimatedImage { false };
    bool isAnimating { false };
    bool canShowAnimationControls { false };
    bool isPausedVideo { false };
    bool isElement { false };
    bool isContentEditable { false };
    Markable<WebCore::ScrollingNodeID> containerScrollingNodeID;
#if ENABLE(DATA_DETECTION)
    bool isDataDetectorLink { false };
#endif
    bool preventTextInteraction { false };
    bool elementContainsImageOverlay { false };
    bool isImageOverlayText { false };
#if ENABLE(SPATIAL_IMAGE_DETECTION)
    bool isSpatialImage { false };
#endif
    bool isInPlugin { false };
    bool needsPointerTouchCompatibilityQuirk { false };
    WebCore::FloatPoint adjustedPointForNodeRespondingToClickEvents;
    URL url;
    URL imageURL;
    String imageMIMEType;
    String title;
    String idAttribute;
    WebCore::IntRect bounds;
#if PLATFORM(MACCATALYST)
    WebCore::IntRect caretRect;
#endif
    RefPtr<WebCore::ShareableBitmap> image;
    String textBefore;
    String textAfter;

    CursorContext cursorContext;

    WebCore::TextIndicatorData linkIndicator;
#if ENABLE(DATA_DETECTION)
    String dataDetectorIdentifier;
    RetainPtr<NSArray> dataDetectorResults;
    WebCore::IntRect dataDetectorBounds;
#endif

    std::optional<WebCore::ElementContext> elementContext;
    std::optional<WebCore::ElementContext> hostImageOrVideoElementContext;

#if ENABLE(ACCESSIBILITY_ANIMATION_CONTROL)
    Vector<WebCore::ElementAnimationContext> animationsAtPoint;
#endif

    // Copy compatible optional bits forward (for example, if we have a InteractionInformationAtPosition
    // with snapshots in it, and perform another request for the same point without requesting the snapshots,
    // we can fetch the cheap information and copy the snapshots into the new response).
    void mergeCompatibleOptionalInformation(const InteractionInformationAtPosition& oldInformation);

    bool isSelectable() const { return selectability == Selectability::Selectable; }
};

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
