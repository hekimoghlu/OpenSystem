/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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

#include "CSSPropertyParser.h"
#include "ImmutableStyleProperties.h"
#include "MutableStyleProperties.h"

namespace WebCore {

inline StyleProperties::StyleProperties(CSSParserMode mode)
    : m_cssParserMode(mode)
    , m_isMutable(true)
{
}

inline StyleProperties::StyleProperties(CSSParserMode mode, unsigned immutableArraySize)
    : m_cssParserMode(mode)
    , m_isMutable(false)
    , m_arraySize(immutableArraySize)
{
}

inline StyleProperties::PropertyReference StyleProperties::propertyAt(unsigned index) const
{
    if (m_isMutable)
        return uncheckedDowncast<MutableStyleProperties>(*this).propertyAt(index);
    return uncheckedDowncast<ImmutableStyleProperties>(*this).propertyAt(index);
}

inline unsigned StyleProperties::propertyCount() const
{
    if (m_isMutable)
        return uncheckedDowncast<MutableStyleProperties>(*this).propertyCount();
    return uncheckedDowncast<ImmutableStyleProperties>(*this).propertyCount();
}

inline void StyleProperties::deref() const
{
    if (!derefBase())
        return;

    if (auto* mutableProperties = dynamicDowncast<MutableStyleProperties>(*this))
        delete mutableProperties;
    else if (auto* immutableProperties = dynamicDowncast<ImmutableStyleProperties>(*this))
        delete immutableProperties;
    else
        RELEASE_ASSERT_NOT_REACHED();
}

inline int StyleProperties::findPropertyIndex(CSSPropertyID propertyID) const
{
    if (m_isMutable)
        return uncheckedDowncast<MutableStyleProperties>(*this).findPropertyIndex(propertyID);
    return uncheckedDowncast<ImmutableStyleProperties>(*this).findPropertyIndex(propertyID);
}

inline int StyleProperties::findCustomPropertyIndex(StringView propertyName) const
{
    if (m_isMutable)
        return uncheckedDowncast<MutableStyleProperties>(*this).findCustomPropertyIndex(propertyName);
    return uncheckedDowncast<ImmutableStyleProperties>(*this).findCustomPropertyIndex(propertyName);
}

inline bool StyleProperties::isEmpty() const
{
    return !propertyCount();
}

inline unsigned StyleProperties::size() const
{
    return propertyCount();
}

inline String serializeLonghandValue(CSSPropertyID property, const CSSValue* value)
{
    return value ? serializeLonghandValue(property, *value) : String();
}

inline CSSValueID longhandValueID(CSSPropertyID property, const CSSValue& value)
{
    return value.isImplicitInitialValue() ? initialValueIDForLonghand(property) : valueID(value);
}

inline std::optional<CSSValueID> longhandValueID(CSSPropertyID property, const CSSValue* value)
{
    if (!value)
        return std::nullopt;
    return longhandValueID(property, *value);
}

}
