/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
#include "MutationObserver.h"
#include <wtf/CheckedRef.h>
#include <wtf/RobinHoodHashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {
class MutationObserverRegistration;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MutationObserverRegistration> : std::true_type { };
}

namespace JSC {
class AbstractSlotVisitor;
}

namespace WebCore {

class QualifiedName;

class MutationObserverRegistration : public CanMakeWeakPtr<MutationObserverRegistration> {
    WTF_MAKE_TZONE_ALLOCATED(MutationObserverRegistration);
public:
    MutationObserverRegistration(MutationObserver&, Node&, MutationObserverOptions, const MemoryCompactLookupOnlyRobinHoodHashSet<AtomString>& attributeFilter);
    ~MutationObserverRegistration();

    void resetObservation(MutationObserverOptions, const MemoryCompactLookupOnlyRobinHoodHashSet<AtomString>& attributeFilter);
    void observedSubtreeNodeWillDetach(Node&);
    UncheckedKeyHashSet<GCReachableRef<Node>> takeTransientRegistrations();
    bool hasTransientRegistrations() const { return !m_transientRegistrationNodes.isEmpty(); }

    bool shouldReceiveMutationFrom(Node&, MutationObserverOptionType, const QualifiedName* attributeName) const;
    bool isSubtree() const { return m_options.contains(MutationObserverOptionType::Subtree); }

    MutationObserver& observer() { return m_observer.get(); }
    Ref<MutationObserver> protectedObserver() { return m_observer; }
    Node& node() { return m_node; }
    MutationRecordDeliveryOptions deliveryOptions() const { return m_options & MutationObserver::AllDeliveryFlags; }
    MutationObserverOptions mutationTypes() const { return m_options & MutationObserver::AllMutationTypes; }

    bool isReachableFromOpaqueRoots(JSC::AbstractSlotVisitor&) const;

private:
    Ref<MutationObserver> m_observer;
    WeakRef<Node, WeakPtrImplWithEventTargetData> m_node;
    RefPtr<Node> m_nodeKeptAlive;
    UncheckedKeyHashSet<GCReachableRef<Node>> m_transientRegistrationNodes;
    MutationObserverOptions m_options;
    MemoryCompactLookupOnlyRobinHoodHashSet<AtomString> m_attributeFilter;
};

} // namespace WebCore
