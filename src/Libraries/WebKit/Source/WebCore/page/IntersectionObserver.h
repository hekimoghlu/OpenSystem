/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include "GCReachableRef.h"
#include "IntersectionObserverCallback.h"
#include "LengthBox.h"
#include "ReducedResolutionSeconds.h"
#include <variant>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class AbstractSlotVisitor;

}

namespace WebCore {

class ContainerNode;
class Document;
class Element;
class IntersectionObserverEntry;

struct IntersectionObserverRegistration {
    WeakPtr<IntersectionObserver> observer;
    std::optional<size_t> previousThresholdIndex;
};

struct IntersectionObserverData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    // IntersectionObservers for which the node that owns this IntersectionObserverData is the root.
    // An IntersectionObserver is only owned by a JavaScript wrapper. ActiveDOMObject::virtualHasPendingActivity
    // is overridden to keep this wrapper alive while the observer has ongoing observations.
    Vector<WeakPtr<IntersectionObserver>> observers;

    // IntersectionObserverRegistrations for which the node that owns this IntersectionObserverData is the target.
    Vector<IntersectionObserverRegistration> registrations;
};

class IntersectionObserver : public RefCountedAndCanMakeWeakPtr<IntersectionObserver> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IntersectionObserver);
public:
    struct Init {
        std::optional<std::variant<RefPtr<Element>, RefPtr<Document>>> root;
        String rootMargin;
        std::variant<double, Vector<double>> threshold;
    };

    static ExceptionOr<Ref<IntersectionObserver>> create(Document&, Ref<IntersectionObserverCallback>&&, Init&&);

    ~IntersectionObserver();

    Document* trackingDocument() const { return m_root ? &m_root->document() : m_implicitRootDocument.get(); }

    ContainerNode* root() const { return m_root.get(); }
    String rootMargin() const;
    const LengthBox& rootMarginBox() const { return m_rootMargin; }
    const Vector<double>& thresholds() const { return m_thresholds; }
    const Vector<WeakPtr<Element, WeakPtrImplWithEventTargetData>>& observationTargets() const { return m_observationTargets; }
    bool hasObservationTargets() const { return m_observationTargets.size(); }
    bool isObserving(const Element&) const;

    void observe(Element&);
    void unobserve(Element&);
    void disconnect();

    struct TakenRecords {
        Vector<Ref<IntersectionObserverEntry>> records;
        Vector<GCReachableRef<Element>> pendingTargets;
    };
    TakenRecords takeRecords();

    void targetDestroyed(Element&);
    void rootDestroyed();

    enum class NeedNotify : bool { No, Yes };
    NeedNotify updateObservations(Document&);

    std::optional<ReducedResolutionSeconds> nowTimestamp() const;

    void appendQueuedEntry(Ref<IntersectionObserverEntry>&&);
    void notify();

    IntersectionObserverCallback* callbackConcurrently() { return m_callback.get(); }
    bool isReachableFromOpaqueRoots(JSC::AbstractSlotVisitor&) const;

private:
    IntersectionObserver(Document&, Ref<IntersectionObserverCallback>&&, ContainerNode* root, LengthBox&& parsedRootMargin, Vector<double>&& thresholds);

    bool removeTargetRegistration(Element&);
    void removeAllTargets();

    struct IntersectionObservationState {
        FloatRect rootBounds;
        std::optional<FloatRect> absoluteIntersectionRect; // Only computed if intersecting.
        std::optional<FloatRect> absoluteTargetRect; // Only computed if first observation, or intersecting.
        std::optional<FloatRect> absoluteRootBounds; // Only computed if observationChanged.
        float intersectionRatio { 0 };
        size_t thresholdIndex { 0 };
        bool canComputeIntersection { false };
        bool isIntersecting { false };
        bool observationChanged { false };
    };

    IntersectionObservationState computeIntersectionState(const IntersectionObserverRegistration&, LocalFrameView&, Element& target, bool applyRootMargin) const;

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_implicitRootDocument;
    WeakPtr<ContainerNode, WeakPtrImplWithEventTargetData> m_root;
    LengthBox m_rootMargin;
    Vector<double> m_thresholds;
    RefPtr<IntersectionObserverCallback> m_callback;
    Vector<WeakPtr<Element, WeakPtrImplWithEventTargetData>> m_observationTargets;
    Vector<GCReachableRef<Element>> m_pendingTargets;
    Vector<Ref<IntersectionObserverEntry>> m_queuedEntries;
    Vector<GCReachableRef<Element>> m_targetsWaitingForFirstObservation;
};


} // namespace WebCore
