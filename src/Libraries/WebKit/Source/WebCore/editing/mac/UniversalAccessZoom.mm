/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#import "config.h"
#import "UniversalAccessZoom.h"

#import "FloatRect.h"
#import "PlatformScreen.h"

namespace WebCore {

void changeUniversalAccessZoomFocus(const IntRect& viewRect, const IntRect& selectionRect)
{
#if PLATFORM(MAC)
    if (!UAZoomEnabled())
        return;

    auto cgCaretRect = NSRectToCGRect(toUserSpaceForPrimaryScreen(selectionRect));
    auto cgViewRect = NSRectToCGRect(toUserSpaceForPrimaryScreen(viewRect));
    
    UAZoomChangeFocus(&cgViewRect, &cgCaretRect, kUAZoomFocusTypeInsertionPoint);
#else
    UNUSED_PARAM(viewRect);
    UNUSED_PARAM(selectionRect);
#endif
}

}
