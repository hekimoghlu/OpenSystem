/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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

#include "AccessibilitySVGObject.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class AccessibilitySVGRoot final : public AccessibilitySVGObject {
public:
    static Ref<AccessibilitySVGRoot> create(AXID, RenderObject&, AXObjectCache*);
    virtual ~AccessibilitySVGRoot();

    void setParent(AccessibilityRenderObject*);
    bool hasAccessibleContent() const;
private:
    explicit AccessibilitySVGRoot(AXID, RenderObject&, AXObjectCache*);

    AccessibilityObject* parentObject() const final;
    bool isAccessibilitySVGRoot() const final { return true; }

    AccessibilityRole determineAccessibilityRole() final;

    WeakPtr<AccessibilityRenderObject> m_parent;
};

} // namespace WebCore 

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilitySVGRoot) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAccessibilitySVGRoot(); } \
SPECIALIZE_TYPE_TRAITS_END()
