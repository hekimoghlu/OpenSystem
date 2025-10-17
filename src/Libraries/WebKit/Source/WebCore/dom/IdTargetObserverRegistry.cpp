/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#include "IdTargetObserverRegistry.h"

#include "IdTargetObserver.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IdTargetObserverRegistry);

IdTargetObserverRegistry::IdTargetObserverRegistry() = default;

IdTargetObserverRegistry::~IdTargetObserverRegistry() = default;

void IdTargetObserverRegistry::addObserver(const AtomString& id, IdTargetObserver& observer)
{
    if (id.isEmpty())
        return;
    
    IdToObserverSetMap::AddResult result = m_registry.ensure(id, [] {
        return makeUnique<ObserverSet>();
    });

    result.iterator->value->observers.add(observer);
}

void IdTargetObserverRegistry::removeObserver(const AtomString& id, IdTargetObserver& observer)
{
    if (id.isEmpty() || m_registry.isEmpty())
        return;

    IdToObserverSetMap::iterator iter = m_registry.find(id);

    CheckedPtr set = iter->value.get();
    set->observers.remove(observer);
    if (set->observers.isEmpty() && set != m_notifyingObserversInSet) {
        set = nullptr;
        m_registry.remove(iter);
    }
}

void IdTargetObserverRegistry::notifyObserversInternal(Element& element, const AtomString& id)
{
    ASSERT(!m_registry.isEmpty());

    m_notifyingObserversInSet = m_registry.get(id);
    if (!m_notifyingObserversInSet)
        return;

    for (auto& observer : copyToVector(m_notifyingObserversInSet->observers)) {
        if (m_notifyingObserversInSet->observers.contains(observer))
            observer->idTargetChanged(element);
    }

    bool hasRemainingObservers = !m_notifyingObserversInSet->observers.isEmpty();
    m_notifyingObserversInSet = nullptr;

    if (!hasRemainingObservers)
        m_registry.remove(id);
}

IdTargetObserverRegistry::ObserverSet::ObserverSet() = default;

IdTargetObserverRegistry::ObserverSet::~ObserverSet() = default;

} // namespace WebCore
