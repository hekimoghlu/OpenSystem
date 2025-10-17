/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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

#include "CSSPropertyNames.h"
#include "CSSValue.h"
#include "WritingMode.h"
#include <wtf/BitSet.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class CSSValueList;

enum class IsImportant : bool { No, Yes };

struct StylePropertyMetadata {
    StylePropertyMetadata(CSSPropertyID propertyID, bool isSetFromShorthand, int indexInShorthandsVector, IsImportant important, bool implicit, bool inherited)
        : m_propertyID(propertyID)
        , m_isSetFromShorthand(isSetFromShorthand)
        , m_indexInShorthandsVector(indexInShorthandsVector)
        , m_important(important == IsImportant::Yes)
        , m_implicit(implicit)
        , m_inherited(inherited)
    {
        ASSERT(propertyID != CSSPropertyInvalid);
        ASSERT_WITH_MESSAGE(propertyID < firstShorthandProperty, "unexpected property: %d", propertyID);
    }

    CSSPropertyID shorthandID() const;
    
    friend bool operator==(const StylePropertyMetadata&, const StylePropertyMetadata&) = default;

    unsigned m_propertyID : 10;
    unsigned m_isSetFromShorthand : 1;
    unsigned m_indexInShorthandsVector : 2; // If this property was set as part of an ambiguous shorthand, gives the index in the shorthands vector.
    unsigned m_important : 1;
    unsigned m_implicit : 1; // Whether or not the property was set implicitly as the result of a shorthand.
    unsigned m_inherited : 1;
};

class CSSProperty {
public:
    CSSProperty(CSSPropertyID propertyID, RefPtr<CSSValue>&& value, IsImportant important = IsImportant::No, bool isSetFromShorthand = false, int indexInShorthandsVector = 0, bool implicit = false)
        : m_metadata(propertyID, isSetFromShorthand, indexInShorthandsVector, important, implicit, isInheritedProperty(propertyID))
        , m_value(WTFMove(value))
    {
    }

    CSSPropertyID id() const { return static_cast<CSSPropertyID>(m_metadata.m_propertyID); }
    bool isSetFromShorthand() const { return m_metadata.m_isSetFromShorthand; };
    CSSPropertyID shorthandID() const { return m_metadata.shorthandID(); };
    bool isImportant() const { return m_metadata.m_important; }

    CSSValue* value() const { return m_value.get(); }
    RefPtr<CSSValue> protectedValue() const { return m_value; }

    static CSSPropertyID resolveDirectionAwareProperty(CSSPropertyID, WritingMode);
    static CSSPropertyID unresolvePhysicalProperty(CSSPropertyID, WritingMode);
    static bool isInheritedProperty(CSSPropertyID);
    static Vector<String> aliasesForProperty(CSSPropertyID);
    static bool isDirectionAwareProperty(CSSPropertyID);
    static bool isInLogicalPropertyGroup(CSSPropertyID);
    static bool areInSameLogicalPropertyGroupWithDifferentMappingLogic(CSSPropertyID, CSSPropertyID);
    static bool isDescriptorOnly(CSSPropertyID);
    static UChar listValuedPropertySeparator(CSSPropertyID);
    static bool isListValuedProperty(CSSPropertyID propertyID) { return !!listValuedPropertySeparator(propertyID); }
    static bool allowsNumberOrIntegerInput(CSSPropertyID);

    // FIXME: Generate from logical property groups.

    // Check if a property is an inset property, as defined in:
    // https://drafts.csswg.org/css-logical-1/#inset-properties
    static bool isInsetProperty(CSSPropertyID);

    // Check if a property is a margin property, as defined in:
    // https://drafts.csswg.org/css-box-4/#margin-properties
    static bool isMarginProperty(CSSPropertyID);

    // Check if a property is a sizing property, as defined in:
    // https://drafts.csswg.org/css-sizing-3/#sizing-property
    static bool isSizingProperty(CSSPropertyID);

    static bool disablesNativeAppearance(CSSPropertyID);

    const StylePropertyMetadata& metadata() const { return m_metadata; }
    static bool isColorProperty(CSSPropertyID propertyId)
    {
        return colorProperties.get(propertyId);
    }

    static const WEBCORE_EXPORT WTF::BitSet<numCSSProperties> colorProperties;
    static const WEBCORE_EXPORT WTF::BitSet<numCSSProperties> physicalProperties;

    bool operator==(const CSSProperty& other) const
    {
        if (!(m_metadata == other.m_metadata))
            return false;

        if (!m_value && !other.m_value)
            return true;

        if (!m_value || !other.m_value)
            return false;
        
        return m_value->equals(*other.m_value);
    }

private:
    StylePropertyMetadata m_metadata;
    RefPtr<CSSValue> m_value;
};

typedef Vector<CSSProperty, 256> ParsedPropertyVector;

} // namespace WebCore

namespace WTF {
template <> struct VectorTraits<WebCore::CSSProperty> : VectorTraitsBase<false, WebCore::CSSProperty> {
    static const bool canInitializeWithMemset = true;
    static const bool canMoveWithMemcpy = true;
};
}
