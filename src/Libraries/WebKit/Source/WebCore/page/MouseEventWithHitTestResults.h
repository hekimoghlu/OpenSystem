/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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

#include "HitTestResult.h"
#include "PlatformMouseEvent.h"

namespace WebCore {

class Scrollbar;

class MouseEventWithHitTestResults {
public:
    MouseEventWithHitTestResults(const PlatformMouseEvent&, const HitTestResult&);

    const PlatformMouseEvent& event() const { return m_event; }
    const HitTestResult& hitTestResult() const { return m_hitTestResult; }
    LayoutPoint localPoint() const { return m_hitTestResult.localPoint(); }
    Scrollbar* scrollbar() const { return m_hitTestResult.scrollbar(); }
    bool isOverLink() const;
    bool isOverWidget() const { return m_hitTestResult.isOverWidget(); }
    Node* targetNode() const { return m_hitTestResult.targetNode(); }
    RefPtr<Node> protectedTargetNode() const  { return m_hitTestResult.protectedTargetNode(); }

private:
    PlatformMouseEvent m_event;
    HitTestResult m_hitTestResult;
};

} // namespace WebCore
