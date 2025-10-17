/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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

#include "ScrollTypes.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS CALayer;
OBJC_CLASS NSScrollerImp;
OBJC_CLASS WebScrollerImpDelegateMac;

namespace WebCore {

class FloatPoint;
class ScrollerPairMac;

class ScrollerMac final : public CanMakeCheckedPtr<ScrollerMac> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ScrollerMac);
    friend class ScrollerPairMac;
public:
    ScrollerMac(ScrollerPairMac&, ScrollbarOrientation);

    ~ScrollerMac();

    void attach();

    ScrollerPairMac& pair() { return m_pair; }

    ScrollbarOrientation orientation() const { return m_orientation; }

    CALayer *hostLayer() const { return m_hostLayer.get(); }
    void setHostLayer(CALayer *);

    RetainPtr<NSScrollerImp> takeScrollerImp() { return std::exchange(m_scrollerImp, { }); }
    NSScrollerImp *scrollerImp() { return m_scrollerImp.get(); }
    void setScrollerImp(NSScrollerImp *imp);
    void updateScrollbarStyle();
    void updatePairScrollerImps();

    void setHiddenByStyle(NativeScrollbarVisibility);

    void updateValues();
    
    String scrollbarState() const;
    
    void mouseEnteredScrollbar();
    void mouseExitedScrollbar();    
    void setLastKnownMousePositionInScrollbar(IntPoint position) { m_lastKnownMousePositionInScrollbar = position; }
    IntPoint lastKnownMousePositionInScrollbar() const;
    void visibilityChanged(bool);
    void updateMinimumKnobLength(int);
    void detach();
    void setEnabled(bool flag) { m_isEnabled = flag; }
    void setScrollbarLayoutDirection(UserInterfaceLayoutDirection);

    void setNeedsDisplay();

private:
    int m_minimumKnobLength { 0 };

    bool m_isEnabled { false };
    bool m_isVisible { false };
    bool m_isHiddenByStyle { false };

    ScrollerPairMac& m_pair;
    const ScrollbarOrientation m_orientation;
    IntPoint m_lastKnownMousePositionInScrollbar;

    RetainPtr<CALayer> m_hostLayer;
    RetainPtr<NSScrollerImp> m_scrollerImp;
    RetainPtr<WebScrollerImpDelegateMac> m_scrollerImpDelegate;
};

}

#endif
