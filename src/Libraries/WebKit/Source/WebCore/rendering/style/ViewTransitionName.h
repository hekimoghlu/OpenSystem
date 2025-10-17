/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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

#include "StyleScopeOrdinal.h"

namespace WebCore::Style {

class ViewTransitionName {
public:
    enum class Type : uint8_t {
        None,
        Auto,
        MatchElement,
        CustomIdent,
    };

    static ViewTransitionName createWithNone()
    {
        return ViewTransitionName(Type::None);
    }

    static ViewTransitionName createWithAuto(ScopeOrdinal ordinal)
    {
        return ViewTransitionName(Type::Auto, ordinal);
    }

    static ViewTransitionName createWithMatchElement(ScopeOrdinal ordinal)
    {
        return ViewTransitionName(Type::MatchElement, ordinal);
    }

    static ViewTransitionName createWithCustomIdent(ScopeOrdinal ordinal, AtomString ident)
    {
        return ViewTransitionName(ordinal, ident);
    }

    bool isNone() const
    {
        return m_type == Type::None;
    }

    bool isAuto() const
    {
        return m_type == Type::Auto;
    }

    bool isMatchElement() const
    {
        return m_type == Type::MatchElement;
    }

    bool isCustomIdent() const
    {
        return m_type == Type::CustomIdent;
    }

    AtomString customIdent() const
    {
        ASSERT(isCustomIdent());
        return m_customIdent;
    }

    ScopeOrdinal scopeOrdinal() const
    {
        ASSERT(!isNone());
        return m_scopeOrdinal;
    }

    bool operator==(const ViewTransitionName& other) const = default;
private:
    Type m_type;
    ScopeOrdinal m_scopeOrdinal;
    AtomString m_customIdent;

    ViewTransitionName(Type type, ScopeOrdinal scopeOrdinal = ScopeOrdinal::Element)
        : m_type(type)
        , m_scopeOrdinal(scopeOrdinal)
    {
    }

    ViewTransitionName(ScopeOrdinal scopeOrdinal, AtomString ident)
        : m_type(Type::CustomIdent)
        , m_scopeOrdinal(scopeOrdinal)
        , m_customIdent(ident)
    {
    }

};

inline TextStream& operator<<(TextStream& ts, const ViewTransitionName& name)
{
    if (name.isAuto())
        ts << "auto"_s;
    else if (name.isMatchElement())
        ts << "match-element"_s;
    else if (name.isNone())
        ts << "none"_s;
    else
        ts << name.customIdent();
    return ts;
}

} // namespace WebCore::Style
