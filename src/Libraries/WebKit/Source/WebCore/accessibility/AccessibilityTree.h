/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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

// This class is representative of role="tree" elements, not the abstract concept
// of the "accessibility tree".
class AccessibilityTree final : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityTree> create(AXID, RenderObject&);
    static Ref<AccessibilityTree> create(AXID, Node&);
    virtual ~AccessibilityTree();

private:
    explicit AccessibilityTree(AXID, RenderObject&);
    explicit AccessibilityTree(AXID, Node&);
    bool computeIsIgnored() const final;
    AccessibilityRole determineAccessibilityRole() final;
    bool isTreeValid() const;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ACCESSIBILITY(AccessibilityTree, isTree())
