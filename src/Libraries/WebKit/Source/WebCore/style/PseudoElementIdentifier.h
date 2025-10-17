/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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

#include "RenderStyleConstants.h"
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>
#include <wtf/text/TextStream.h>

namespace WebCore::Style {

struct PseudoElementIdentifier {
    PseudoId pseudoId;

    // highlight name for ::highlight or view transition name for view transition pseudo elements.
    AtomString nameArgument { nullAtom() };

    friend bool operator==(const PseudoElementIdentifier& a, const PseudoElementIdentifier& b) = default;
};

inline void add(Hasher& hasher, const PseudoElementIdentifier& pseudoElementIdentifier)
{
    add(hasher, pseudoElementIdentifier.pseudoId, pseudoElementIdentifier.nameArgument);
}

inline WTF::TextStream& operator<<(WTF::TextStream& ts, const PseudoElementIdentifier& pseudoElementIdentifier)
{
    ts << "::" << pseudoElementIdentifier.pseudoId;
    if (!pseudoElementIdentifier.nameArgument.isNull())
        ts << "(" << pseudoElementIdentifier.nameArgument << ")";
    return ts;
}

inline bool isNamedViewTransitionPseudoElement(const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier)
{
    if (!pseudoElementIdentifier)
        return false;

    switch (pseudoElementIdentifier->pseudoId) {
    case PseudoId::ViewTransitionGroup:
    case PseudoId::ViewTransitionImagePair:
    case PseudoId::ViewTransitionOld:
    case PseudoId::ViewTransitionNew:
        return true;
    default:
        return false;
    }
};

} // namespace WebCore

namespace WTF {

template<>
struct HashTraits<WebCore::Style::PseudoElementIdentifier> : GenericHashTraits<WebCore::Style::PseudoElementIdentifier> {
    typedef WebCore::Style::PseudoElementIdentifier EmptyValueType;

    static constexpr bool emptyValueIsZero = false;
    static EmptyValueType emptyValue() { return WebCore::Style::PseudoElementIdentifier { WebCore::PseudoId::AfterLastInternalPseudoId, nullAtom() }; }

    static void constructDeletedValue(WebCore::Style::PseudoElementIdentifier& pseudoElementIdentifier) { pseudoElementIdentifier = WebCore::Style::PseudoElementIdentifier { WebCore::PseudoId::None, nullAtom() }; }
    static bool isDeletedValue(const WebCore::Style::PseudoElementIdentifier& pseudoElementIdentifier) { return pseudoElementIdentifier == WebCore::Style::PseudoElementIdentifier { WebCore::PseudoId::None, nullAtom() }; }
};

template<>
struct DefaultHash<WebCore::Style::PseudoElementIdentifier> {
    static unsigned hash(const WebCore::Style::PseudoElementIdentifier& data) { return computeHash(data); }
    static bool equal(const WebCore::Style::PseudoElementIdentifier& a, const WebCore::Style::PseudoElementIdentifier& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

template<>
struct HashTraits<std::optional<WebCore::Style::PseudoElementIdentifier>> : GenericHashTraits<std::optional<WebCore::Style::PseudoElementIdentifier>> {
    typedef std::optional<WebCore::Style::PseudoElementIdentifier> EmptyValueType;

    static constexpr bool emptyValueIsZero = false;
    static EmptyValueType emptyValue() { return WebCore::Style::PseudoElementIdentifier { WebCore::PseudoId::AfterLastInternalPseudoId, nullAtom() }; }

    static void constructDeletedValue(std::optional<WebCore::Style::PseudoElementIdentifier>& pseudoElementIdentifier) { pseudoElementIdentifier = WebCore::Style::PseudoElementIdentifier { WebCore::PseudoId::None, nullAtom() }; }
    static bool isDeletedValue(const std::optional<WebCore::Style::PseudoElementIdentifier>& pseudoElementIdentifier) { return pseudoElementIdentifier == WebCore::Style::PseudoElementIdentifier { WebCore::PseudoId::None, nullAtom() }; }
};

template<>
struct DefaultHash<std::optional<WebCore::Style::PseudoElementIdentifier>> {
    static unsigned hash(const std::optional<WebCore::Style::PseudoElementIdentifier>& data) { return computeHash(data); }
    static bool equal(const std::optional<WebCore::Style::PseudoElementIdentifier>& a, const std::optional<WebCore::Style::PseudoElementIdentifier>& b) { return a == b; }

    static const bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace WTF
