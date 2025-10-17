/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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

#include "CSSSelector.h"
#include <wtf/BloomFilter.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Element;

namespace Style {

enum class MatchElement : uint8_t;

class HasSelectorFilter {
    WTF_MAKE_TZONE_ALLOCATED(HasSelectorFilter);
public:
    enum class Type : uint8_t { Children, Descendants };
    HasSelectorFilter(const Element&, Type);

    Type type() const { return m_type; }
    static std::optional<Type> typeForMatchElement(MatchElement);

    using Key = unsigned;
    static Key makeKey(const CSSSelector& hasSelector);

    bool reject(const CSSSelector& hasSelector) const { return reject(makeKey(hasSelector)); }
    bool reject(Key key) const { return key && !m_filter.mayContain(key); }

private:
    void add(const Element&);

    const Type m_type;
    BloomFilter<12> m_filter;
};

}
}
