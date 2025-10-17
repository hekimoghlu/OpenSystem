/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
#include "CustomStateSet.h"

#include "CSSParser.h"
#include "Element.h"
#include "PseudoClassChangeInvalidation.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CustomStateSet);

bool CustomStateSet::addToSetLike(const AtomString& state)
{
    std::optional<Style::PseudoClassChangeInvalidation> styleInvalidation;
    if (RefPtr element = m_element.get())
        styleInvalidation.emplace(*element, CSSSelector::PseudoClass::State, Style::PseudoClassChangeInvalidation::AnyValue);

    return m_states.add(AtomString(state)).isNewEntry;
}

bool CustomStateSet::removeFromSetLike(const AtomString& state)
{
    std::optional<Style::PseudoClassChangeInvalidation> styleInvalidation;
    if (RefPtr element = m_element.get())
        styleInvalidation.emplace(*element, CSSSelector::PseudoClass::State, Style::PseudoClassChangeInvalidation::AnyValue);

    return m_states.remove(AtomString(state));
}

void CustomStateSet::clearFromSetLike()
{
    std::optional<Style::PseudoClassChangeInvalidation> styleInvalidation;
    if (RefPtr element = m_element.get())
        styleInvalidation.emplace(*element, CSSSelector::PseudoClass::State, Style::PseudoClassChangeInvalidation::AnyValue);

    m_states.clear();
}

bool CustomStateSet::has(const AtomString& state) const
{
    return m_states.contains(state);
}

} // namespace WebCore
