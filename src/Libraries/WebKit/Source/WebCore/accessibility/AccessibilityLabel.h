/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

class AccessibilityLabel final : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityLabel> create(AXID, RenderObject&);
    virtual ~AccessibilityLabel();

    bool containsOnlyStaticText() const;
private:
    explicit AccessibilityLabel(AXID, RenderObject&);
    bool computeIsIgnored() const final { return isIgnoredByDefault(); }

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::Label; }

    bool isAccessibilityLabelInstance() const final { return true; }
    String stringValue() const final;
    void addChildren() final;
    void clearChildren() final;
    mutable bool m_containsOnlyStaticTextDirty : 1;
    mutable bool m_containsOnlyStaticText : 1;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityLabel) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAccessibilityLabelInstance(); } \
SPECIALIZE_TYPE_TRAITS_END()
