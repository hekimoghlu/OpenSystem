/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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

#include <wtf/EnumTraits.h>

namespace WebCore {
namespace Style {

// This is used to identify style scopes that can affect an element.
// Scopes are in tree-of-trees order. Styles from earlier scopes win over later ones (modulo !important).
enum class ScopeOrdinal : int8_t {
    ContainingHostLimit = std::numeric_limits<int8_t>::min(),
    ContainingHost = -1, // ::part rules and author-exposed UA pseudo classes from the host tree scope. Values less than ContainingHost indicate enclosing scopes.
    Element = 0, // Normal rules in the same tree where the element is.
    FirstSlot = 1, // ::slotted rules in the parent's shadow tree. Values greater than FirstSlot indicate subsequent slots in the chain.
    SlotLimit = std::numeric_limits<int8_t>::max() - 1,
    Shadow = std::numeric_limits<int8_t>::max(), // :host rules in element's own shadow tree.
};

inline ScopeOrdinal& operator++(ScopeOrdinal& ordinal)
{
    ASSERT(ordinal < ScopeOrdinal::SlotLimit);
    return ordinal = static_cast<ScopeOrdinal>(enumToUnderlyingType(ordinal) + 1);
}

inline ScopeOrdinal& operator--(ScopeOrdinal& ordinal)
{
    ASSERT(ordinal > ScopeOrdinal::ContainingHostLimit);
    return ordinal = static_cast<ScopeOrdinal>(enumToUnderlyingType(ordinal) - 1);
}

}
}
