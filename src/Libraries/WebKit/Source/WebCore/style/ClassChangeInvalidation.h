/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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

#include "Element.h"
#include "StyleInvalidator.h"
#include <wtf/Vector.h>

namespace WebCore {

class SpaceSplitString;

namespace Style {

class ClassChangeInvalidation {
public:
    ClassChangeInvalidation(Element&, const SpaceSplitString& oldClasses, const SpaceSplitString& newClasses);
    ~ClassChangeInvalidation();

private:
    void computeInvalidation(const SpaceSplitString& oldClasses, const SpaceSplitString& newClasses);
    void invalidateBeforeChange();
    void invalidateAfterChange();

    const bool m_isEnabled;
    Element& m_element;

    Invalidator::MatchElementRuleSets m_beforeChangeRuleSets;
    Invalidator::MatchElementRuleSets m_afterChangeRuleSets;
};

inline ClassChangeInvalidation::ClassChangeInvalidation(Element& element, const SpaceSplitString& oldClasses, const SpaceSplitString& newClasses)
    : m_isEnabled(element.needsStyleInvalidation())
    , m_element(element)

{
    if (!m_isEnabled)
        return;
    computeInvalidation(oldClasses, newClasses);
    invalidateBeforeChange();
}

inline ClassChangeInvalidation::~ClassChangeInvalidation()
{
    if (!m_isEnabled)
        return;
    invalidateAfterChange();
}

}
}
