/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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

#if PLATFORM(COCOA)
#include "DrawingArea.h"

#include <QuartzCore/QuartzCore.h>
#include <wtf/BlockObjCExceptions.h>

namespace WebKit {

RetainPtr<CABasicAnimation> DrawingArea::transientZoomSnapAnimationForKeyPath(ASCIILiteral keyPath)
{
    const float transientZoomSnapBackDuration = 0.25;
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    
    RetainPtr<CABasicAnimation> animation = [CABasicAnimation animationWithKeyPath:keyPath.createNSString().get()];
    [animation setDuration:transientZoomSnapBackDuration];
    [animation setFillMode:kCAFillModeForwards];
    [animation setRemovedOnCompletion:false];
    [animation setAdditive:NO];
    [animation setTimingFunction:[CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseInEaseOut]];
    return animation;
    
    END_BLOCK_OBJC_EXCEPTIONS
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
