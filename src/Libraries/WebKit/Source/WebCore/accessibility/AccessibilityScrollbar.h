/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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

class Scrollbar;

class AccessibilityScrollbar final : public AccessibilityMockObject {
public:
    static Ref<AccessibilityScrollbar> create(AXID, Scrollbar&);

private:
    explicit AccessibilityScrollbar(AXID, Scrollbar&);

    bool canSetValueAttribute() const final { return true; }

    bool isAccessibilityScrollbar() const final { return true; }
    LayoutRect elementRect() const final;

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::ScrollBar; }
    AccessibilityOrientation orientation() const final;
    Document* document() const final;
    bool isEnabled() const final;

    // Assumes float [0..1]
    bool setValue(float) final;
    float valueForRange() const final;

    Ref<Scrollbar> m_scrollbar;
};

} // namespace WebCore


SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityScrollbar) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAccessibilityScrollbar(); } \
SPECIALIZE_TYPE_TRAITS_END()
