/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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

#include "AccessibilityMockObject.h"

namespace WebCore {
    
class RenderTableSection;

class AccessibilityTableColumn final : public AccessibilityMockObject {
public:
    static Ref<AccessibilityTableColumn> create(AXID);
    virtual ~AccessibilityTableColumn();

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::Column; }

    void setColumnIndex(unsigned);
    unsigned columnIndex() const final { return m_columnIndex; }

    void addChildren() final;
    void setParent(AccessibilityObject*) final;

    LayoutRect elementRect() const final;

private:
    explicit AccessibilityTableColumn(AXID);
    
    bool computeIsIgnored() const final;

    bool isAccessibilityTableColumnInstance() const final { return true; }
    unsigned m_columnIndex;
};

} // namespace WebCore 

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityTableColumn) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAccessibilityTableColumnInstance(); } \
SPECIALIZE_TYPE_TRAITS_END()
