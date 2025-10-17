/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#include "CSSValueKeywords.h"
#include <wtf/Vector.h>

namespace WebCore {

class StylePropertyShorthand {
public:
    StylePropertyShorthand() = default;

    template<std::size_t numProperties> StylePropertyShorthand(CSSPropertyID id, std::span<const CSSPropertyID, numProperties> properties)
        : m_properties(properties.data())
        , m_length(properties.size())
        , m_shorthandID(id)
    {
        static_assert(numProperties != std::dynamic_extent);
    }

    const CSSPropertyID* begin() const { return std::to_address(properties().begin()); }
    const CSSPropertyID* end() const { return std::to_address(properties().end()); }

    size_t length() const { return m_length; }
    CSSPropertyID id() const { return m_shorthandID; }

    std::span<const CSSPropertyID> properties() const { return unsafeMakeSpan(m_properties, m_length); }

private:
    const CSSPropertyID* m_properties { nullptr };
    unsigned m_length { 0 };
    CSSPropertyID m_shorthandID { CSSPropertyInvalid };
};

// Custom StylePropertyShorthand function.
StylePropertyShorthand transitionShorthandForParsing();

// Returns empty value if the property is not a shorthand.
// The implementation is generated in StylePropertyShorthandFunctions.cpp.
StylePropertyShorthand shorthandForProperty(CSSPropertyID);

// Return the list of shorthands for a given longhand.
// The implementation is generated in StylePropertyShorthandFunctions.cpp.
using StylePropertyShorthandVector = Vector<StylePropertyShorthand, 4>;
StylePropertyShorthandVector matchingShorthandsForLonghand(CSSPropertyID);

unsigned indexOfShorthandForLonghand(CSSPropertyID, const StylePropertyShorthandVector&);

} // namespace WebCore

namespace WTF {
template<> inline size_t containerSize(const WebCore::StylePropertyShorthand& container) { return container.length(); }
}
