/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#include "Element.h"
#include "StyleInvalidator.h"
#include <wtf/Vector.h>

namespace WebCore {
namespace Style {

class PseudoClassChangeInvalidation {
public:
    PseudoClassChangeInvalidation(Element&, CSSSelector::PseudoClass, bool value, InvalidationScope = InvalidationScope::All);
    enum AnyValueTag { AnyValue };
    PseudoClassChangeInvalidation(Element&, CSSSelector::PseudoClass, AnyValueTag);
    PseudoClassChangeInvalidation(Element&, std::initializer_list<std::pair<CSSSelector::PseudoClass, bool>>);

    ~PseudoClassChangeInvalidation();

private:
    enum class Value : uint8_t { False, True, Any };
    void computeInvalidation(CSSSelector::PseudoClass, Value, Style::InvalidationScope);
    void collectRuleSets(const PseudoClassInvalidationKey&, Value, InvalidationScope);
    void invalidateBeforeChange();
    void invalidateAfterChange();

    const bool m_isEnabled;
    Element& m_element;

    Invalidator::MatchElementRuleSets m_beforeChangeRuleSets;
    Invalidator::MatchElementRuleSets m_afterChangeRuleSets;
};

Vector<PseudoClassInvalidationKey, 4> makePseudoClassInvalidationKeys(CSSSelector::PseudoClass, const Element&);

inline void emplace(std::optional<PseudoClassChangeInvalidation>& invalidation, Element& element, std::initializer_list<std::pair<CSSSelector::PseudoClass, bool>> pseudoClasses)
{
    invalidation.emplace(element, pseudoClasses);
}

inline PseudoClassChangeInvalidation::PseudoClassChangeInvalidation(Element& element, CSSSelector::PseudoClass pseudoClass, bool value, Style::InvalidationScope invalidationScope)
    : m_isEnabled(element.needsStyleInvalidation())
    , m_element(element)

{
    if (!m_isEnabled)
        return;
    computeInvalidation(pseudoClass, value ? Value::True : Value::False, invalidationScope);
    invalidateBeforeChange();
}

inline PseudoClassChangeInvalidation::PseudoClassChangeInvalidation(Element& element, CSSSelector::PseudoClass pseudoClass, AnyValueTag)
    : m_isEnabled(element.needsStyleInvalidation())
    , m_element(element)

{
    if (!m_isEnabled)
        return;
    computeInvalidation(pseudoClass, Value::Any, InvalidationScope::All);
    invalidateBeforeChange();
}

inline PseudoClassChangeInvalidation::PseudoClassChangeInvalidation(Element& element, std::initializer_list<std::pair<CSSSelector::PseudoClass, bool>> pseudoClasses)
    : m_isEnabled(element.needsStyleInvalidation())
    , m_element(element)
{
    if (!m_isEnabled)
        return;
    for (auto pseudoClass : pseudoClasses)
        computeInvalidation(pseudoClass.first, pseudoClass.second ? Value::True : Value::False, Style::InvalidationScope::All);
    invalidateBeforeChange();
}

inline PseudoClassChangeInvalidation::~PseudoClassChangeInvalidation()
{
    if (!m_isEnabled)
        return;
    invalidateAfterChange();
}

}
}
