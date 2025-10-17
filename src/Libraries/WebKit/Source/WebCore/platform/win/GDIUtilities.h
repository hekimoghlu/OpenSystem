/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#ifndef GDIUtilties_h
#define GDIUtilties_h

#include "IntPoint.h"

#include <windows.h>

namespace WebCore {

WEBCORE_EXPORT float deviceScaleFactorForWindow(HWND);

inline LPARAM makeScaledPoint(IntPoint point, float scaleFactor)
{
    float inverseScaleFactor = 1.0f / scaleFactor;
    point.scale(inverseScaleFactor, inverseScaleFactor);
    return MAKELPARAM(point.x(), point.y());
}

inline unsigned short buttonsForEvent(WPARAM wparam)
{
    unsigned short buttons = 0;
    if (wparam & MK_LBUTTON)
        buttons |= 1;
    if (wparam & MK_MBUTTON)
        buttons |= 4;
    if (wparam & MK_RBUTTON)
        buttons |= 2;
    return buttons;
}

inline LONG getDoubleClickTime()
{
    // GetDoubleClickTime() returns 0 in the non-interactive window station on Windows 10 version 2004
    LONG doubleClickTime = GetDoubleClickTime();
    return doubleClickTime ? doubleClickTime : 500;
}

} // namespace WebCore

#endif // GDIUtilties_h
