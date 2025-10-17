/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "ContentExtensionActions.h"
#include "ResourceLoadInfo.h"
#include <wtf/Hasher.h>
#include <wtf/text/WTFString.h>

namespace WebCore::ContentExtensions {

// A ContentExtensionRule is the smallest unit in a ContentExtension.
//
// It is composed of a trigger and an action. The trigger defines on what kind of content this extension should apply.
// The action defines what to perform on that content.

struct Trigger {
    String urlFilter;
    bool urlFilterIsCaseSensitive { false };
    bool topURLFilterIsCaseSensitive { false };
    bool frameURLFilterIsCaseSensitive { false };
    ResourceFlags flags { 0 };
    Vector<String> conditions;

    WEBCORE_EXPORT Trigger isolatedCopy() const &;
    WEBCORE_EXPORT Trigger isolatedCopy() &&;
    
    void checkValidity()
    {
        auto actionCondition = static_cast<ActionCondition>(flags & ActionConditionMask);
        ASSERT_UNUSED(actionCondition, conditions.isEmpty() == (actionCondition == ActionCondition::None));
        if (topURLFilterIsCaseSensitive)
            ASSERT(actionCondition == ActionCondition::IfTopURL || actionCondition == ActionCondition::UnlessTopURL);
        if (frameURLFilterIsCaseSensitive)
            ASSERT(actionCondition == ActionCondition::IfFrameURL || actionCondition == ActionCondition::UnlessFrameURL);
    }

    bool isEmpty() const
    {
        return urlFilter.isEmpty()
            && !urlFilterIsCaseSensitive
            && !topURLFilterIsCaseSensitive
            && !frameURLFilterIsCaseSensitive
            && !flags
            && conditions.isEmpty();
    }

    friend bool operator==(const Trigger&, const Trigger&) = default;
};

inline void add(Hasher& hasher, const Trigger& trigger)
{
    add(hasher, trigger.urlFilterIsCaseSensitive, trigger.urlFilter, trigger.flags, trigger.conditions);
}

struct TriggerHash {
    static unsigned hash(const Trigger& trigger)
    {
        return computeHash(trigger);
    }
    static bool equal(const Trigger& a, const Trigger& b)
    {
        return a == b;
    }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

struct TriggerHashTraits : public WTF::CustomHashTraits<Trigger> {
    static constexpr bool emptyValueIsZero = false;
    static constexpr bool hasIsEmptyValueFunction = true;

    static void constructDeletedValue(Trigger& trigger)
    {
        new (NotNull, std::addressof(trigger.urlFilter)) String(WTF::HashTableDeletedValue);
    }

    static bool isDeletedValue(const Trigger& trigger)
    {
        return trigger.urlFilter.isHashTableDeletedValue();
    }

    static Trigger emptyValue()
    {
        return Trigger();
    }

    static bool isEmptyValue(const Trigger& trigger)
    {
        return trigger.isEmpty();
    }
};

struct Action {
    Action(ActionData&& data)
        : m_data(WTFMove(data)) { }

    friend bool operator==(const Action&, const Action&) = default;

    const ActionData& data() const { return m_data; }

    WEBCORE_EXPORT Action isolatedCopy() const &;
    WEBCORE_EXPORT Action isolatedCopy() &&;

private:
    ActionData m_data;
};

struct DeserializedAction : public Action {
    static DeserializedAction deserialize(std::span<const uint8_t>, uint32_t location);
    static size_t serializedLength(std::span<const uint8_t>, uint32_t location);

    uint32_t actionID() const { return m_actionID; }

private:
    DeserializedAction(uint32_t actionID, ActionData&& data)
        : Action(WTFMove(data))
        , m_actionID(actionID) { }

    const uint32_t m_actionID;
};

class ContentExtensionRule {
public:
    WEBCORE_EXPORT ContentExtensionRule(Trigger&&, Action&&);

    const Trigger& trigger() const { return m_trigger; }
    const Action& action() const { return m_action; }

    ContentExtensionRule isolatedCopy() const & { return { m_trigger.isolatedCopy(), m_action.isolatedCopy() }; }
    ContentExtensionRule isolatedCopy() && { return { WTFMove(m_trigger).isolatedCopy(), WTFMove(m_action).isolatedCopy() }; }
    friend bool operator==(const ContentExtensionRule&, const ContentExtensionRule&) = default;

private:
    Trigger m_trigger;
    Action m_action;
};

} // namespace WebCore::ContentExtensions

#endif // ENABLE(CONTENT_EXTENSIONS)
