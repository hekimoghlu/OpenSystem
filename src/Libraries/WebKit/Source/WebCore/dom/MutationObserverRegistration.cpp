/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#include "config.h"

#include "MutationObserverRegistration.h"

#include "Document.h"
#include "JSNodeCustom.h"
#include "QualifiedName.h"
#include "WebCoreOpaqueRootInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MutationObserverRegistration);

MutationObserverRegistration::MutationObserverRegistration(MutationObserver& observer, Node& node, MutationObserverOptions options, const MemoryCompactLookupOnlyRobinHoodHashSet<AtomString>& attributeFilter)
    : m_observer(observer)
    , m_node(node)
    , m_options(options)
    , m_attributeFilter(attributeFilter)
{
    protectedObserver()->observationStarted(*this);
}

MutationObserverRegistration::~MutationObserverRegistration()
{
    takeTransientRegistrations();
    protectedObserver()->observationEnded(*this);
}

void MutationObserverRegistration::resetObservation(MutationObserverOptions options, const MemoryCompactLookupOnlyRobinHoodHashSet<AtomString>& attributeFilter)
{
    takeTransientRegistrations();
    m_options = options;
    m_attributeFilter = attributeFilter;
}

void MutationObserverRegistration::observedSubtreeNodeWillDetach(Node& node)
{
    if (!isSubtree())
        return;

    node.registerTransientMutationObserver(*this);
    m_observer->setHasTransientRegistration(node.protectedDocument());

    if (m_transientRegistrationNodes.isEmpty()) {
        ASSERT(!m_nodeKeptAlive);
        m_nodeKeptAlive = m_node.ptr(); // Balanced in takeTransientRegistrations.
    }
    m_transientRegistrationNodes.add(node);
}

UncheckedKeyHashSet<GCReachableRef<Node>> MutationObserverRegistration::takeTransientRegistrations()
{
    if (m_transientRegistrationNodes.isEmpty()) {
        ASSERT(!m_nodeKeptAlive);
        return { };
    }

    for (auto& node : m_transientRegistrationNodes)
        node->unregisterTransientMutationObserver(*this);

    auto returnValue = std::exchange(m_transientRegistrationNodes, { });

    ASSERT(m_nodeKeptAlive);
    m_nodeKeptAlive = nullptr; // Balanced in observeSubtreeNodeWillDetach.

    return returnValue;
}

bool MutationObserverRegistration::shouldReceiveMutationFrom(Node& node, MutationObserverOptionType type, const QualifiedName* attributeName) const
{
    ASSERT((type == MutationObserverOptionType::Attributes && attributeName) || !attributeName);
    if (!m_options.contains(type))
        return false;

    if (m_node.ptr() != &node && !isSubtree())
        return false;

    if (type != MutationObserverOptionType::Attributes || !m_options.contains(MutationObserverOptionType::AttributeFilter))
        return true;

    if (!attributeName->namespaceURI().isNull())
        return false;

    return m_attributeFilter.contains(attributeName->localName());
}

bool MutationObserverRegistration::isReachableFromOpaqueRoots(JSC::AbstractSlotVisitor& visitor) const
{
    if (containsWebCoreOpaqueRoot(visitor, m_node.ptr()))
        return true;

    for (auto& node : m_transientRegistrationNodes) {
        if (containsWebCoreOpaqueRoot(visitor, node.get()))
            return true;
    }

    return false;
}

} // namespace WebCore
