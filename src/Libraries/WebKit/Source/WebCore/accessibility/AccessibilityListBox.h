/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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

#include "AccessibilityRenderObject.h"

namespace WebCore {

class AccessibilityListBox final : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityListBox> create(AXID, RenderObject&);
    virtual ~AccessibilityListBox();

    WEBCORE_EXPORT void setSelectedChildren(const AccessibilityChildrenVector&) final;

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::ListBox; }

    AccessibilityChildrenVector visibleChildren() final;
    
    void addChildren() final;

private:
    explicit AccessibilityListBox(AXID, RenderObject&);

    bool isAccessibilityListBoxInstance() const final { return true; }
    AccessibilityObject* listBoxOptionAccessibilityObject(HTMLElement*) const;
    AccessibilityObject* elementAccessibilityHitTest(const IntPoint&) const final;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityListBox) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAccessibilityListBoxInstance(); } \
SPECIALIZE_TYPE_TRAITS_END()
