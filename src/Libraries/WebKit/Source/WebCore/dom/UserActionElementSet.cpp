/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
#include "UserActionElementSet.h"

#include "Element.h"

namespace WebCore {

void UserActionElementSet::clear()
{
    for (auto iterator = m_elements.begin(); iterator != m_elements.end(); ++iterator)
        iterator->key.setUserActionElement(false);
    m_elements.clear();
}

bool UserActionElementSet::hasFlag(const Element& element, Flag flag) const
{
    // Caller has the responsibility to check isUserActionElement before calling this.
    ASSERT(element.isUserActionElement());

    return m_elements.get(element).contains(flag);
}

void UserActionElementSet::clearFlags(Element& element, OptionSet<Flag> flags)
{
    ASSERT(!flags.isEmpty());

    if (!element.isUserActionElement())
        return;

    auto iterator = m_elements.find(element);
    ASSERT(iterator != m_elements.end());
    auto updatedFlags = iterator->value - flags;
    if (updatedFlags.isEmpty()) {
        element.setUserActionElement(false);
        m_elements.remove(iterator);
    } else
        iterator->value = updatedFlags;
}

void UserActionElementSet::setFlags(Element& element, OptionSet<Flag> flags)
{
    ASSERT(!flags.isEmpty());

    m_elements.ensure(element, [] {
        return OptionSet<Flag>();
    }).iterator->value.add(flags);

    element.setUserActionElement(true);
}

}
