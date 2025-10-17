/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
    
class AccessibilityTreeItem final : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityTreeItem> create(AXID, RenderObject&);
    static Ref<AccessibilityTreeItem> create(AXID, Node&);
    virtual ~AccessibilityTreeItem();

    bool supportsCheckedState() const final;

private:
    explicit AccessibilityTreeItem(AXID, RenderObject&);
    explicit AccessibilityTreeItem(AXID, Node&);
    bool shouldIgnoreAttributeRole() const final { return !m_isTreeItemValid; }
    AccessibilityRole determineAccessibilityRole() final;
    bool m_isTreeItemValid;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ACCESSIBILITY(AccessibilityTreeItem, isTreeItem())
