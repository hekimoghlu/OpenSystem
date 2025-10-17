/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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

#include "ScrollbarsController.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// A Mock implementation of ScrollbarsController used to test the scroll events
// received by the scrollbar controller. Tests can enable this mock object using
// the internal setting setMockScrollbarsControllerEnabled().

class ScrollbarsControllerMock final : public ScrollbarsController {
    WTF_MAKE_TZONE_ALLOCATED(ScrollbarsControllerMock);
    WTF_MAKE_NONCOPYABLE(ScrollbarsControllerMock);
public:
    ScrollbarsControllerMock(ScrollableArea&, Function<void(const String&)>&&);
    virtual ~ScrollbarsControllerMock();
    bool isScrollbarsControllerMock() const final { return true; }

private:

    void didAddVerticalScrollbar(Scrollbar*) final;
    void didAddHorizontalScrollbar(Scrollbar*) final;
    void willRemoveVerticalScrollbar(Scrollbar*) final;
    void willRemoveHorizontalScrollbar(Scrollbar*) final;
    void mouseEnteredContentArea() final;
    void mouseMovedInContentArea() final;
    void mouseExitedContentArea() final;
    void mouseEnteredScrollbar(Scrollbar*) const final;
    void mouseExitedScrollbar(Scrollbar*) const final;
    void mouseIsDownInScrollbar(Scrollbar*, bool) const final;
    ASCIILiteral scrollbarPrefix(Scrollbar*) const;

    Function<void(const String&)> m_logger;
    Scrollbar* m_verticalScrollbar { nullptr };
    Scrollbar* m_horizontalScrollbar { nullptr };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ScrollbarsControllerMock)
    static bool isType(const WebCore::ScrollbarsController& controller) { return controller.isScrollbarsControllerMock(); }
SPECIALIZE_TYPE_TRAITS_END()
