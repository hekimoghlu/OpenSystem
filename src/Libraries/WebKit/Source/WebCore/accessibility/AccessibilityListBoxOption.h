/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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

#include "AccessibilityNodeObject.h"

namespace WebCore {

class HTMLElement;
class HTMLSelectElement;

class AccessibilityListBoxOption final : public AccessibilityNodeObject {
public:
    static Ref<AccessibilityListBoxOption> create(AXID, HTMLElement&);
    virtual ~AccessibilityListBoxOption();

    bool isSelected() const final;
    void setSelected(bool) final;

private:
    explicit AccessibilityListBoxOption(AXID, HTMLElement&);

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::ListBoxOption; }
    bool isEnabled() const final;
    bool isSelectedOptionActive() const final;
    String stringValue() const final;
    Element* actionElement() const final;
    bool canSetSelectedAttribute() const final;

    LayoutRect elementRect() const final;
    AccessibilityObject* parentObject() const final;

    bool isAccessibilityListBoxOptionInstance() const final { return true; }
    bool canHaveChildren() const final { return false; }
    HTMLSelectElement* listBoxOptionParentNode() const;
    int listBoxOptionIndex() const;
    IntRect listBoxOptionRect() const;
    AccessibilityObject* listBoxOptionAccessibilityObject(HTMLElement*) const;
    bool computeIsIgnored() const final;
};

} // namespace WebCore 

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityListBoxOption) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAccessibilityListBoxOptionInstance(); } \
SPECIALIZE_TYPE_TRAITS_END()
