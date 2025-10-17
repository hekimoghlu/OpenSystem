/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include "ImmutableStyleProperties.h"

#include "CSSCustomPropertyValue.h"
#include "StylePropertiesInlines.h"
#include <wtf/HashMap.h>
#include <wtf/Hasher.h>
#include <wtf/IndexedRange.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ImmutableStyleProperties);

ImmutableStyleProperties::ImmutableStyleProperties(std::span<const CSSProperty> properties, CSSParserMode mode)
    : StyleProperties(mode, properties.size())
{
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    auto* metadataArray = const_cast<StylePropertyMetadata*>(this->metadataArray());
    auto* valueArray = std::bit_cast<PackedPtr<CSSValue>*>(this->valueArray());
    for (auto [i, property] : indexedRange(properties)) {
        metadataArray[i] = property.metadata();
        RefPtr value = property.value();
        valueArray[i] = value.get();
        value->ref();
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

ImmutableStyleProperties::~ImmutableStyleProperties()
{
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    auto* valueArray = std::bit_cast<PackedPtr<CSSValue>*>(this->valueArray());
    for (unsigned i = 0; i < m_arraySize; ++i)
        valueArray[i]->deref();
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

Ref<ImmutableStyleProperties> ImmutableStyleProperties::create(std::span<const CSSProperty> properties, CSSParserMode mode)
{
    void* slot = ImmutableStylePropertiesMalloc::malloc(objectSize(properties.size()));
    return adoptRef(*new (NotNull, slot) ImmutableStyleProperties(properties, mode));
}

static auto& deduplicationMap()
{
    static NeverDestroyed<UncheckedKeyHashMap<unsigned, Ref<ImmutableStyleProperties>, AlreadyHashed>> map;
    return map.get();
}

Ref<ImmutableStyleProperties> ImmutableStyleProperties::createDeduplicating(std::span<const CSSProperty> properties, CSSParserMode mode)
{
    static constexpr auto maximumDeduplicationMapSize = 1024u;
    if (deduplicationMap().size() >= maximumDeduplicationMapSize)
        deduplicationMap().remove(deduplicationMap().random());

    auto computeHash = [&] {
        Hasher hasher;
        add(hasher, mode);
        for (auto& property : properties) {
            if (!property.value()->addHash(hasher))
                return 0u;
            add(hasher, property.id(), property.isImportant());
        }
        return hasher.hash();
    };

    auto hash = computeHash();
    if (!hash)
        return create(properties, mode);

    auto result = deduplicationMap().ensure(hash, [&] {
        return create(properties, mode);
    });

    auto isEqual = [&](auto& existingValue) {
        if (existingValue.propertyCount() != properties.size())
            return false;
        if (existingValue.cssParserMode() != mode)
            return false;
        for (auto [i, property] : indexedRange(properties)) {
            if (existingValue.propertyAt(i).toCSSProperty() != property)
                return false;
        }
        return true;
    };

    if (!result.isNewEntry && !isEqual(result.iterator->value.get()))
        return create(properties, mode);

    return result.iterator->value;
}

void ImmutableStyleProperties::clearDeduplicationMap()
{
    deduplicationMap().clear();
}

int ImmutableStyleProperties::findPropertyIndex(CSSPropertyID propertyID) const
{
    // Convert here propertyID into an uint16_t to compare it with the metadata's m_propertyID to avoid
    // the compiler converting it to an int multiple times in the loop.
    uint16_t id = enumToUnderlyingType(propertyID);
    for (int n = m_arraySize - 1 ; n >= 0; --n) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        if (metadataArray()[n].m_propertyID == id)
            return n;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }
    return -1;
}

int ImmutableStyleProperties::findCustomPropertyIndex(StringView propertyName) const
{
    for (int n = m_arraySize - 1 ; n >= 0; --n) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        if (metadataArray()[n].m_propertyID == CSSPropertyCustom) {
            // We found a custom property. See if the name matches.
            auto* value = valueArray()[n].get();
            if (!value)
                continue;
            if (downcast<CSSCustomPropertyValue>(*value).name() == propertyName)
                return n;
        }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }
    return -1;
}

}
