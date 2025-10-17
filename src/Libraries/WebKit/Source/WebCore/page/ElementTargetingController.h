/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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

#include "Document.h"
#include "ElementIdentifier.h"
#include "ElementTargetingTypes.h"
#include "EventTarget.h"
#include "IntRectHash.h"
#include "Region.h"
#include "ScriptExecutionContextIdentifier.h"
#include "Timer.h"
#include <wtf/ApproximateTime.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Element;
class Document;
class Image;
class Node;
class Page;

class ElementTargetingController final : public CanMakeCheckedPtr<ElementTargetingController> {
    WTF_MAKE_TZONE_ALLOCATED(ElementTargetingController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ElementTargetingController);
public:
    ElementTargetingController(Page&);

    WEBCORE_EXPORT Vector<TargetedElementInfo> findTargets(TargetedElementRequest&&);
    WEBCORE_EXPORT Vector<Vector<TargetedElementInfo>> findAllTargets(float);

    WEBCORE_EXPORT bool adjustVisibility(Vector<TargetedElementAdjustment>&&);
    void adjustVisibilityInRepeatedlyTargetedRegions(Document&);

    void reset();
    void didChangeMainDocument(Document* newDocument);

    WEBCORE_EXPORT uint64_t numberOfVisibilityAdjustmentRects();
    WEBCORE_EXPORT bool resetVisibilityAdjustments(const Vector<TargetedElementIdentifiers>&);

    WEBCORE_EXPORT RefPtr<Image> snapshotIgnoringVisibilityAdjustment(ElementIdentifier, ScriptExecutionContextIdentifier);

private:
    void cleanUpAdjustmentClientRects();

    void applyVisibilityAdjustmentFromSelectors();

    struct FindElementFromSelectorsResult {
        RefPtr<Element> element;
        String lastSelectorIncludingPseudo;
    };
    FindElementFromSelectorsResult findElementFromSelectors(const TargetedElementSelectors&);

    RefPtr<Document> mainDocument() const;

    void dispatchVisibilityAdjustmentStateDidChange();
    void selectorBasedVisibilityAdjustmentTimerFired();

    std::pair<Vector<Ref<Node>>, RefPtr<Element>> findNodes(FloatPoint location, bool shouldIgnorePointerEventsNone);
    std::pair<Vector<Ref<Node>>, RefPtr<Element>> findNodes(const String& searchText);
    std::pair<Vector<Ref<Node>>, RefPtr<Element>> findNodes(const TargetedElementSelectors&);

    enum class IncludeNearbyElements : bool { No, Yes };
    enum class CheckViewportAreaRatio : bool { No, Yes };
    Vector<TargetedElementInfo> extractTargets(Vector<Ref<Node>>&&, RefPtr<Element>&& innerElement, CheckViewportAreaRatio, IncludeNearbyElements);

    void recomputeAdjustedElementsIfNeeded();

    void topologicallySortElementsHelper(ElementIdentifier currentElementID, Vector<ElementIdentifier>& depthSortedIDs, UncheckedKeyHashSet<ElementIdentifier>& processingIDs, UncheckedKeyHashSet<ElementIdentifier>& unprocessedIDs, const UncheckedKeyHashMap<ElementIdentifier, UncheckedKeyHashSet<ElementIdentifier>>& elementIDToOccludedElementIDs);
    Vector<ElementIdentifier> topologicallySortElements(const UncheckedKeyHashMap<ElementIdentifier, UncheckedKeyHashSet<ElementIdentifier>>& elementIDToOccludedElementIDs);

    WeakPtr<Page> m_page;
    DeferrableOneShotTimer m_recentAdjustmentClientRectsCleanUpTimer;
    WeakHashSet<Document, WeakPtrImplWithEventTargetData> m_documentsAffectedByVisibilityAdjustment;
    UncheckedKeyHashMap<ElementIdentifier, IntRect> m_recentAdjustmentClientRects;
    ApproximateTime m_startTimeForSelectorBasedVisibilityAdjustment;
    Timer m_selectorBasedVisibilityAdjustmentTimer;
    Vector<std::pair<Markable<ElementIdentifier>, TargetedElementSelectors>> m_visibilityAdjustmentSelectors;
    Vector<TargetedElementSelectors> m_initialVisibilityAdjustmentSelectors;
    Region m_adjustmentClientRegion;
    Region m_repeatedAdjustmentClientRegion;
    WeakHashSet<Element, WeakPtrImplWithEventTargetData> m_adjustedElements;
    FloatSize m_viewportSizeForVisibilityAdjustment;
    unsigned m_additionalAdjustmentCount { 0 };
    bool m_didCollectInitialAdjustments { false };
    bool m_shouldRecomputeAdjustedElements { false };
};

} // namespace WebCore
