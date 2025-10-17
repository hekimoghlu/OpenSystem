/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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

#include "StyleProperties.h"

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ImmutableStyleProperties);
class ImmutableStyleProperties final : public StyleProperties {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(ImmutableStyleProperties);
public:
    inline void deref() const;

    WEBCORE_EXPORT ~ImmutableStyleProperties();
    static Ref<ImmutableStyleProperties> create(std::span<const CSSProperty> properties, CSSParserMode);
    static Ref<ImmutableStyleProperties> createDeduplicating(std::span<const CSSProperty> properties, CSSParserMode);

    unsigned propertyCount() const { return m_arraySize; }
    bool isEmpty() const { return !propertyCount(); }
    PropertyReference propertyAt(unsigned index) const;

    Iterator<ImmutableStyleProperties> begin() const { return { *this }; }
    static constexpr std::nullptr_t end() { return nullptr; }
    unsigned size() const { return propertyCount(); }

    int findPropertyIndex(CSSPropertyID) const;
    int findCustomPropertyIndex(StringView propertyName) const;

    static constexpr size_t objectSize(unsigned propertyCount);

    static void clearDeduplicationMap();

    void* m_storage;

private:
    PackedPtr<const CSSValue>* valueArray() const;
    const StylePropertyMetadata* metadataArray() const;
    ImmutableStyleProperties(std::span<const CSSProperty>, CSSParserMode);
};

inline void ImmutableStyleProperties::deref() const
{
    if (derefBase())
        delete this;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
inline PackedPtr<const CSSValue>* ImmutableStyleProperties::valueArray() const
{
    return std::bit_cast<PackedPtr<const CSSValue>*>(std::bit_cast<const uint8_t*>(metadataArray()) + (m_arraySize * sizeof(StylePropertyMetadata)));
}
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

inline const StylePropertyMetadata* ImmutableStyleProperties::metadataArray() const
{
    return reinterpret_cast<const StylePropertyMetadata*>(const_cast<const void**>((&(this->m_storage))));
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
inline ImmutableStyleProperties::PropertyReference ImmutableStyleProperties::propertyAt(unsigned index) const
{
    return PropertyReference(metadataArray()[index], valueArray()[index].get());
}

constexpr size_t ImmutableStyleProperties::objectSize(unsigned count)
{
    return sizeof(ImmutableStyleProperties) - sizeof(void*) + sizeof(StylePropertyMetadata) * count + sizeof(PackedPtr<const CSSValue>) * count;
}
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ImmutableStyleProperties)
    static bool isType(const WebCore::StyleProperties& properties) { return !properties.isMutable(); }
SPECIALIZE_TYPE_TRAITS_END()
