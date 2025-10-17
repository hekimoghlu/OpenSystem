/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#include "StyleCustomPropertyData.h"

namespace WebCore {

static constexpr auto maximumAncestorCount = 4;

StyleCustomPropertyData::StyleCustomPropertyData(const StyleCustomPropertyData& other)
    : m_size(other.m_size)
    , m_mayHaveAnimatableProperties(other.m_mayHaveAnimatableProperties)
{
    auto shouldReferenceAsParentValues = [&] {
        // Always reference the root style since it likely gets shared a lot.
        if (!other.m_parentValues && !other.m_ownValues.isEmpty())
            return true;

        // Limit the list length.
        if (other.m_ancestorCount >= maximumAncestorCount)
            return false;

        // If the number of properties is small we just copy them to avoid creating unnecessarily long linked lists.
        static constexpr auto parentReferencePropertyCountThreshold = 8;
        return other.m_ownValues.size() > parentReferencePropertyCountThreshold;
    }();

    // If there are mutations on multiple levels this constructs a linked list of property data objects.
    if (shouldReferenceAsParentValues)
        m_parentValues = &other;
    else {
        m_parentValues = other.m_parentValues;
        m_ownValues = other.m_ownValues;
    }

    if (m_parentValues)
        m_ancestorCount = m_parentValues->m_ancestorCount + 1;

#if ASSERT_ENABLED
    if (m_parentValues)
        m_parentValues->m_hasChildren = true;
#endif
}

const CSSCustomPropertyValue* StyleCustomPropertyData::get(const AtomString& name) const
{
    for (auto* propertyData = this; propertyData; propertyData = propertyData->m_parentValues.get()) {
        if (auto* value = propertyData->m_ownValues.get(name))
            return value;
    }
    return nullptr;
}

void StyleCustomPropertyData::set(const AtomString& name, Ref<const CSSCustomPropertyValue>&& value)
{
    ASSERT(!m_hasChildren);
    ASSERT([&] {
        auto* existing = get(name);
        return !existing || !existing->equals(value);
    }());

    m_mayHaveAnimatableProperties = m_mayHaveAnimatableProperties || value->isAnimatable();

    auto addResult = m_ownValues.set(name, WTFMove(value));

    bool isNewProperty = addResult.isNewEntry && (!m_parentValues || !m_parentValues->get(name));
    if (isNewProperty)
        ++m_size;
}

bool StyleCustomPropertyData::operator==(const StyleCustomPropertyData& other) const
{
    if (m_size != other.m_size)
        return false;

    if (m_parentValues == other.m_parentValues) {
        // This relies on the values in m_ownValues never being equal to those in m_parentValues.
        if (m_ownValues.size() != other.m_ownValues.size())
            return false;

        for (auto& entry : m_ownValues) {
            auto* otherValue = other.m_ownValues.get(entry.key);
            if (!otherValue || !entry.value->equals(*otherValue))
                return false;
        }
        return true;
    }

    bool isEqual = true;
    forEachInternal([&](auto& entry) {
        auto* otherValue = other.get(entry.key);
        if (!otherValue || !entry.value->equals(*otherValue)) {
            isEqual = false;
            return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    });

    return isEqual;
}

template<typename Callback>
void StyleCustomPropertyData::forEachInternal(Callback&& callback) const
{
    Vector<const StyleCustomPropertyData*, maximumAncestorCount> descendants;

    auto isOverridenByDescendants = [&](auto& key) {
        for (auto* descendant : descendants) {
            if (descendant->m_ownValues.contains(key))
                return true;
        }
        return false;
    };

    auto* propertyData = this;
    while (true) {
        for (auto& entry : propertyData->m_ownValues) {
            if (isOverridenByDescendants(entry.key))
                continue;
            auto status = callback(entry);
            if (status == IterationStatus::Done)
                return;
        }
        if (!propertyData->m_parentValues)
            return;
        descendants.append(propertyData);
        propertyData = propertyData->m_parentValues.get();
    }
}

void StyleCustomPropertyData::forEach(const Function<IterationStatus(const KeyValuePair<AtomString, RefPtr<const CSSCustomPropertyValue>>&)>& callback) const
{
    forEachInternal(callback);
}

AtomString StyleCustomPropertyData::findKeyAtIndex(unsigned index) const
{
    unsigned currentIndex = 0;
    AtomString key;
    forEachInternal([&](auto& entry) {
        if (currentIndex == index) {
            key = entry.key;
            return IterationStatus::Done;
        }
        ++currentIndex;
        return IterationStatus::Continue;
    });
    return key;
}

#if !LOG_DISABLED
void StyleCustomPropertyData::dumpDifferences(TextStream& ts, const StyleCustomPropertyData& other) const
{
    if (*this != other)
        ts << "custom properies differ\n";
}
#endif // !LOG_DISABLED

} // namespace WebCore
