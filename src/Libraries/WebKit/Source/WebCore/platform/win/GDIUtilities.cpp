/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
#include "GDIUtilities.h"

#include "HWndDC.h"
#include <wtf/win/SoftLinking.h>

SOFT_LINK_LIBRARY(user32);
SOFT_LINK_OPTIONAL(user32, GetDpiForWindow, UINT, STDAPICALLTYPE, (HWND));

namespace WebCore {

float deviceScaleFactorForWindow(HWND window)
{
    if (window && GetDpiForWindowPtr())
        return GetDpiForWindowPtr()(window) / 96.0f;
    HWndDC dc(window);
    return ::GetDeviceCaps(dc, LOGPIXELSX) / 96.0f;
}

} // namespace WebCore
