/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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

#if PLATFORM(MAC)

#include "Scrollbar.h"

OBJC_CLASS NSScrollerImp;

namespace WebCore {

class ScrollbarMac : public Scrollbar {
public:
    ScrollbarMac(ScrollableArea&, ScrollbarOrientation, ScrollbarWidth);
    ~ScrollbarMac() { }

    NSScrollerImp* scrollerImp() const;
    void createScrollerImp(NSScrollerImp* oldScrollerImp = nullptr);
    bool isMacScrollbar() const override { return true; }

private:
    void updateScrollerImpState();

    RetainPtr<NSScrollerImp> m_scrollerImp;
};

}

SPECIALIZE_TYPE_TRAITS_SCROLLBAR_HOLDS_SCROLLER_IMP(ScrollbarMac, isMacScrollbar())

#endif // PLATFORM(MAC)
