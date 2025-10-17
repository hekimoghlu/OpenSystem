/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

namespace WebCore {

namespace Style {

class IdChangeInvalidation {
public:
    IdChangeInvalidation(Element&, const AtomString& oldId, const AtomString& newId);
    ~IdChangeInvalidation();

private:
    void invalidateStyle(const AtomString&);
    void invalidateStyleWithRuleSets();

    const bool m_isEnabled;
    Element& m_element;

    AtomString m_newId;

    Invalidator::MatchElementRuleSets m_matchElementRuleSets;
};

inline IdChangeInvalidation::IdChangeInvalidation(Element& element, const AtomString& oldId, const AtomString& newId)
    : m_isEnabled(element.needsStyleInvalidation())
    , m_element(element)
{
    if (!m_isEnabled)
        return;
    if (oldId == newId)
        return;
    m_newId = newId;

    invalidateStyle(oldId);
    invalidateStyleWithRuleSets();
}

inline IdChangeInvalidation::~IdChangeInvalidation()
{
    if (!m_isEnabled)
        return;
    invalidateStyle(m_newId);
    invalidateStyleWithRuleSets();
}

}
}
