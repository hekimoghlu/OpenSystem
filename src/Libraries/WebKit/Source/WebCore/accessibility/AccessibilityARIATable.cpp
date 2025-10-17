/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#include "AccessibilityARIATable.h"

#include "AXObjectCache.h"

namespace WebCore {

class RenderObject;

AccessibilityARIATable::AccessibilityARIATable(AXID axID, RenderObject& renderer)
    : AccessibilityTable(axID, renderer)
{
}

AccessibilityARIATable::AccessibilityARIATable(AXID axID, Node& node)
    : AccessibilityTable(axID, node)
{
}

AccessibilityARIATable::~AccessibilityARIATable() = default;

Ref<AccessibilityARIATable> AccessibilityARIATable::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityARIATable(axID, renderer));
}

Ref<AccessibilityARIATable> AccessibilityARIATable::create(AXID axID, Node& node)
{
    return adoptRef(*new AccessibilityARIATable(axID, node));
}

bool AccessibilityARIATable::isMultiSelectable() const
{
    // Per https://w3c.github.io/aria/#table, role="table" elements don't support selection,
    // or aria-multiselectable â€” only role="grid" and role="treegrid".
    if (!hasGridRole())
        return false;

    const AtomString& ariaMultiSelectable = getAttribute(HTMLNames::aria_multiselectableAttr);
    return !equalLettersIgnoringASCIICase(ariaMultiSelectable, "false"_s);
}

} // namespace WebCore
