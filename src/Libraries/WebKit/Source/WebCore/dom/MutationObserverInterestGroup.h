/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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

#include "DocumentInlines.h"
#include "MutationObserver.h"
#include <memory>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class QualifiedName;

class MutationObserverInterestGroup {
    WTF_MAKE_TZONE_ALLOCATED(MutationObserverInterestGroup);
public:
    MutationObserverInterestGroup(UncheckedKeyHashMap<Ref<MutationObserver>, MutationRecordDeliveryOptions>&&, MutationRecordDeliveryOptions);

    static std::unique_ptr<MutationObserverInterestGroup> createForChildListMutation(Node& target)
    {
        if (!target.document().hasMutationObserversOfType(MutationObserverOptionType::ChildList))
            return nullptr;

        MutationRecordDeliveryOptions oldValueFlag;
        return createIfNeeded(target, MutationObserverOptionType::ChildList, oldValueFlag);
    }

    static std::unique_ptr<MutationObserverInterestGroup> createForCharacterDataMutation(Node& target)
    {
        if (!target.document().hasMutationObserversOfType(MutationObserverOptionType::CharacterData))
            return nullptr;

        return createIfNeeded(target, MutationObserverOptionType::CharacterData, MutationObserverOptionType::CharacterDataOldValue);
    }

    static std::unique_ptr<MutationObserverInterestGroup> createForAttributesMutation(Node& target, const QualifiedName& attributeName)
    {
        if (!target.document().hasMutationObserversOfType(MutationObserverOptionType::Attributes))
            return nullptr;

        return createIfNeeded(target, MutationObserverOptionType::Attributes, MutationObserverOptionType::AttributeOldValue, &attributeName);
    }

    bool isOldValueRequested() const;
    void enqueueMutationRecord(Ref<MutationRecord>&&);

private:
    static std::unique_ptr<MutationObserverInterestGroup> createIfNeeded(Node& target, MutationObserverOptionType, MutationRecordDeliveryOptions oldValueFlag, const QualifiedName* attributeName = nullptr);

    bool hasOldValue(MutationRecordDeliveryOptions options) const { return options.containsAny(m_oldValueFlag); }

    UncheckedKeyHashMap<Ref<MutationObserver>, MutationRecordDeliveryOptions> m_observers;
    MutationRecordDeliveryOptions m_oldValueFlag;
};

} // namespace WebCore
