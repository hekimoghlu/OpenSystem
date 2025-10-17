/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

class AccessibilitySVGObject : public AccessibilityRenderObject {
public:
    static Ref<AccessibilitySVGObject> create(AXID, RenderObject&, AXObjectCache*);
    virtual ~AccessibilitySVGObject();

protected:
    explicit AccessibilitySVGObject(AXID, RenderObject&, AXObjectCache*);
    AXObjectCache* axObjectCache() const final { return m_axObjectCache.get(); }
    AccessibilityRole determineAriaRoleAttribute() const final;

private:
    String description() const final;
    String helpText() const final;
    void accessibilityText(Vector<AccessibilityText>&) const final;
    AccessibilityRole determineAccessibilityRole() override;
    bool inheritsPresentationalRole() const final;
    bool computeIsIgnored() const final;

    AccessibilityObject* targetForUseElement() const;

    // Returns true if the SVG element associated with this object has a <title> or <desc> child.
    bool hasTitleOrDescriptionChild() const;
    template <typename ChildrenType>
    Element* childElementWithMatchingLanguage(ChildrenType&) const;

    WeakPtr<AXObjectCache> m_axObjectCache;
};

} // namespace WebCore
