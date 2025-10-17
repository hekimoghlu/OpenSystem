/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

#include "MutationObserverInterestGroup.h"

#include "MutationObserverRegistration.h"
#include "MutationRecord.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MutationObserverInterestGroup);

inline MutationObserverInterestGroup::MutationObserverInterestGroup(UncheckedKeyHashMap<Ref<MutationObserver>, MutationRecordDeliveryOptions>&& observers, MutationRecordDeliveryOptions oldValueFlag)
    : m_observers(WTFMove(observers))
    , m_oldValueFlag(oldValueFlag)
{
    ASSERT(!m_observers.isEmpty());
}

std::unique_ptr<MutationObserverInterestGroup> MutationObserverInterestGroup::createIfNeeded(Node& target, MutationObserverOptionType type, MutationRecordDeliveryOptions oldValueFlag, const QualifiedName* attributeName)
{
    ASSERT((type == MutationObserverOptionType::Attributes && attributeName) || !attributeName);
    auto observers = target.registeredMutationObservers(type, attributeName);
    if (observers.isEmpty())
        return nullptr;

    return makeUnique<MutationObserverInterestGroup>(WTFMove(observers), oldValueFlag);
}

bool MutationObserverInterestGroup::isOldValueRequested() const
{
    for (auto options : m_observers.values()) {
        if (hasOldValue(options))
            return true;
    }
    return false;
}

void MutationObserverInterestGroup::enqueueMutationRecord(Ref<MutationRecord>&& mutation)
{
    RefPtr<MutationRecord> mutationWithNullOldValue;
    for (auto& observerOptionsPair : m_observers) {
        Ref observer = observerOptionsPair.key.get();
        if (hasOldValue(observerOptionsPair.value)) {
            observer->enqueueMutationRecord(mutation.copyRef());
            continue;
        }
        if (!mutationWithNullOldValue) {
            if (mutation->oldValue().isNull())
                mutationWithNullOldValue = mutation.ptr();
            else
                mutationWithNullOldValue = MutationRecord::createWithNullOldValue(mutation).ptr();
        }
        observer->enqueueMutationRecord(*mutationWithNullOldValue);
    }
}

} // namespace WebCore
