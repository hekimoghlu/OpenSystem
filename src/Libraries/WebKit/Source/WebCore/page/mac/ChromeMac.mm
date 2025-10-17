/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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
#import "Chrome.h"

#import "ChromeClient.h"
#import <wtf/BlockObjCExceptions.h>

#if PLATFORM(IOS_FAMILY)
#import "WAKResponder.h"
#import "WAKView.h"
#endif

namespace WebCore {


void Chrome::focusNSView(NSView* view)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS

    // Handle the WK2 case where there is no view passed in.
    if (!view) {
        client().makeFirstResponder();
        return;
    }
    
    NSResponder *firstResponder = client().firstResponder();
    if (firstResponder == view)
        return;

    if (![view window] || ![view superview] || ![view acceptsFirstResponder])
        return;

    client().makeFirstResponder(view);

    // Setting focus can actually cause a style change which might
    // remove the view from its superview while it's being made
    // first responder. This confuses AppKit so we must restore
    // the old first responder.
    if (![view superview])
        client().makeFirstResponder(firstResponder);

    END_BLOCK_OBJC_EXCEPTIONS
}

} // namespace WebCore
