/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

#include <memory>
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class Element;
class IdTargetObserver;

class IdTargetObserverRegistry final : public CanMakeCheckedPtr<IdTargetObserverRegistry> {
    WTF_MAKE_TZONE_ALLOCATED(IdTargetObserverRegistry);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IdTargetObserverRegistry);
    friend class IdTargetObserver;
public:
    IdTargetObserverRegistry();
    ~IdTargetObserverRegistry();

    void notifyObservers(Element&, const AtomString& id);

private:
    void addObserver(const AtomString& id, IdTargetObserver&);
    void removeObserver(const AtomString& id, IdTargetObserver&);
    void notifyObserversInternal(Element&, const AtomString& id);

    struct ObserverSet final : public CanMakeCheckedPtr<ObserverSet> {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        WTF_STRUCT_OVERRIDE_DELETE_FOR_CHECKED_PTR(ObserverSet);

        ObserverSet();
        ~ObserverSet();
        UncheckedKeyHashSet<CheckedRef<IdTargetObserver>> observers;
    };

    using IdToObserverSetMap = UncheckedKeyHashMap<AtomString, std::unique_ptr<ObserverSet>>;
    IdToObserverSetMap m_registry;
    CheckedPtr<ObserverSet> m_notifyingObserversInSet;
};

inline void IdTargetObserverRegistry::notifyObservers(Element& element, const AtomString& id)
{
    ASSERT(!id.isEmpty());
    ASSERT(!m_notifyingObserversInSet);
    if (m_registry.isEmpty())
        return;
    IdTargetObserverRegistry::notifyObserversInternal(element, id);
}

} // namespace WebCore
