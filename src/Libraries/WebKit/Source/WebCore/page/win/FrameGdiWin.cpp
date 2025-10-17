/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#include "config.h"
#include "FrameWin.h"

#include "GraphicsContext.h"
#include "GraphicsContextWin.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include <wtf/win/GDIObject.h>

namespace WebCore {

extern HDC g_screenDC;

GDIObject<HBITMAP> imageFromRect(LocalFrame* frame, IntRect& ir)
{
    int w;
    int h;
    auto* view = frame->view();
    if (view->parent()) {
        ir.setLocation(view->parent()->convertChildToSelf(view, ir.location()));
        w = ir.width() * frame->pageZoomFactor() + 0.5;
        h = ir.height() * frame->pageZoomFactor() + 0.5;
    } else {
        ir = view->contentsToWindow(ir);
        w = ir.width();
        h = ir.height();
    }

    auto bmpDC = adoptGDIObject(::CreateCompatibleDC(g_screenDC));
    auto hBmp = adoptGDIObject(::CreateCompatibleBitmap(g_screenDC, w, h));
    if (!hBmp)
        return nullptr;

    HGDIOBJ hbmpOld = ::SelectObject(bmpDC.get(), hBmp.get());

    {
        GraphicsContextWin gc(bmpDC.get());
        view->paint(&gc, ir);
    }

    ::SelectObject(bmpDC.get(), hbmpOld);

    return hBmp;
}

} // namespace WebCore
