/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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

#include "GCReachableRef.h"
#include "ResizeObservation.h"
#include "ResizeObserverCallback.h"
#include <wtf/Lock.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace JSC {

class AbstractSlotVisitor;

}

namespace WebCore {

class Document;
class Element;
struct ResizeObserverOptions;

struct ResizeObserverData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    Vector<WeakPtr<ResizeObserver>> observers;
};

using NativeResizeObserverCallback = void (*)(const Vector<Ref<ResizeObserverEntry>>&, ResizeObserver&);
using JSOrNativeResizeObserverCallback = std::variant<RefPtr<ResizeObserverCallback>, NativeResizeObserverCallback>;

class ResizeObserver : public RefCountedAndCanMakeWeakPtr<ResizeObserver> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ResizeObserver);
public:
    static Ref<ResizeObserver> create(Document&, Ref<ResizeObserverCallback>&&);
    static Ref<ResizeObserver> createNativeObserver(Document&, NativeResizeObserverCallback&&);
    ~ResizeObserver();

    bool hasObservations() const { return m_observations.size(); }
    bool hasActiveObservations() const { return m_activeObservations.size(); }

    void observe(Element&);
    void observe(Element&, const ResizeObserverOptions&);
    void unobserve(Element&);
    void disconnect();
    void targetDestroyed(Element&);

    static size_t maxElementDepth() { return SIZE_MAX; }
    size_t gatherObservations(size_t depth);
    void deliverObservations();
    bool hasSkippedObservations() const { return m_hasSkippedObservations; }
    void setHasSkippedObservations(bool skipped) { m_hasSkippedObservations = skipped; }

    void resetObservationSize(Element&);

    ResizeObserverCallback* callbackConcurrently();
    bool isReachableFromOpaqueRoots(JSC::AbstractSlotVisitor&) const;

private:
    ResizeObserver(Document&, JSOrNativeResizeObserverCallback&&);

    bool removeTarget(Element&);
    void removeAllTargets();
    bool removeObservation(const Element&);
    void observeInternal(Element&, const ResizeObserverBoxOptions);
    bool isNativeCallback();
    bool isJSCallback();

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    JSOrNativeResizeObserverCallback m_JSOrNativeCallback;
    Vector<Ref<ResizeObservation>> m_observations;

    Vector<Ref<ResizeObservation>> m_activeObservations;
    Vector<GCReachableRef<Element>> m_activeObservationTargets;
    Vector<GCReachableRef<Element>> m_targetsWaitingForFirstObservation;

    bool m_hasSkippedObservations { false };
};

} // namespace WebCore
