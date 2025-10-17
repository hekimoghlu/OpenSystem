/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include "AccessibilityARIAGridCell.h"

#include "AccessibilityObject.h"
#include "AccessibilityTable.h"
#include "HTMLNames.h"

namespace WebCore {
    
using namespace HTMLNames;

AccessibilityARIAGridCell::AccessibilityARIAGridCell(AXID axID, RenderObject& renderer)
    : AccessibilityTableCell(axID, renderer)
{
}

AccessibilityARIAGridCell::AccessibilityARIAGridCell(AXID axID, Node& node)
    : AccessibilityTableCell(axID, node)
{
}

AccessibilityARIAGridCell::~AccessibilityARIAGridCell() = default;

Ref<AccessibilityARIAGridCell> AccessibilityARIAGridCell::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityARIAGridCell(axID, renderer));
}

Ref<AccessibilityARIAGridCell> AccessibilityARIAGridCell::create(AXID axID, Node& node)
{
    return adoptRef(*new AccessibilityARIAGridCell(axID, node));
}

AccessibilityTable* AccessibilityARIAGridCell::parentTable() const
{
    // ARIA gridcells may have multiple levels of unignored ancestors that are not the parent table,
    // including rows and interactive rowgroups. In addition, poorly-formed grids may contain elements
    // which pass the tests for inclusion.
    return dynamicDowncast<AccessibilityTable>(Accessibility::findAncestor<AccessibilityObject>(*this, false, [] (const auto& ancestor) {
        RefPtr ancestorTable = dynamicDowncast<AccessibilityTable>(ancestor);
        return ancestorTable && ancestorTable->isExposable() && !ancestorTable->isIgnored();
    }));
}
    
String AccessibilityARIAGridCell::readOnlyValue() const
{
    if (hasAttribute(aria_readonlyAttr))
        return getAttribute(aria_readonlyAttr).string().convertToASCIILowercase();

    // ARIA 1.1 requires user agents to propagate the grid's aria-readonly value to all
    // gridcell elements if the property is not present on the gridcell element itelf.
    if (RefPtr parent = parentTable())
        return parent->readOnlyValue();

    return String();
}
  
} // namespace WebCore
