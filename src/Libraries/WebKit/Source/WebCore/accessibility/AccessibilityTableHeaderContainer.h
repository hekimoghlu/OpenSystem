/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
#include "AccessibilityTable.h"
#include "IntRect.h"

namespace WebCore {

class AccessibilityTableHeaderContainer final : public AccessibilityMockObject {
public:
    static Ref<AccessibilityTableHeaderContainer> create(AXID axID);
    virtual ~AccessibilityTableHeaderContainer();
    
    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::TableHeaderContainer; }

    void addChildren() final;

    LayoutRect elementRect() const final;

private:
    explicit AccessibilityTableHeaderContainer(AXID);
    
    bool computeIsIgnored() const final;

    LayoutRect m_headerRect;
}; 
    
} // namespace WebCore 
