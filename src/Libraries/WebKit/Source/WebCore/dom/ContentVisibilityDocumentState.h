/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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

#include "IntersectionObserver.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashMap.h>

namespace WebCore {

class Document;
class Element;

enum class DidUpdateAnyContentRelevancy : bool { No, Yes };
enum class IsSkippedContent : bool { No, Yes };
// https://drafts.csswg.org/css-contain/#proximity-to-the-viewport
enum class ViewportProximity : bool { Far, Near };

enum class HadInitialVisibleContentVisibilityDetermination : bool { No, Yes };

class ContentVisibilityDocumentState {
    WTF_MAKE_TZONE_ALLOCATED(ContentVisibilityDocumentState);
public:
    static void observe(Element&);
    static void unobserve(Element&);

    void updateContentRelevancyForScrollIfNeeded(const Element& scrollAnchor);

    bool hasObservationTargets() const { return m_observer && m_observer->hasObservationTargets(); }

    DidUpdateAnyContentRelevancy updateRelevancyOfContentVisibilityElements(OptionSet<ContentRelevancy>) const;
    HadInitialVisibleContentVisibilityDetermination determineInitialVisibleContentVisibility() const;

    void updateViewportProximity(const Element&, ViewportProximity);

    static void updateAnimations(const Element&, IsSkippedContent wasSkipped, IsSkippedContent becomesSkipped);

private:
    bool checkRelevancyOfContentVisibilityElement(Element&, OptionSet<ContentRelevancy>) const;

    void removeViewportProximity(const Element&);

    IntersectionObserver* intersectionObserver(Document&);

    RefPtr<IntersectionObserver> m_observer;

    WeakHashMap<Element, ViewportProximity, WeakPtrImplWithEventTargetData> m_elementViewportProximities;
};

} // namespace
