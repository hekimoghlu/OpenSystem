/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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

#include "ListStyleType.h"
#include "RenderStyleConstants.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class CounterContent {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(CounterContent);
public:
    CounterContent(const AtomString& identifier, ListStyleType style, const AtomString& separator)
        : m_identifier(identifier)
        , m_listStyle(style)
        , m_separator(separator)
    {
        ASSERT(style.type != ListStyleType::Type::String);
    }

    const AtomString& identifier() const { return m_identifier; }
    ListStyleType listStyleType() const { return m_listStyle; }
    const AtomString& separator() const { return m_separator; }

    friend bool operator==(const CounterContent&, const CounterContent&) = default;

private:
    AtomString m_identifier;
    ListStyleType m_listStyle;
    AtomString m_separator;
};

} // namespace WebCore
