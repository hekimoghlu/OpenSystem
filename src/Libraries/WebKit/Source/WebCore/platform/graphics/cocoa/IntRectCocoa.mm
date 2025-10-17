/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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
#import "IntRect.h"

#if !PLATFORM(MAC)
#import <UIKit/UIGeometry.h>
#endif

namespace WebCore {

id makeNSArrayElement(const IntRect& rect)
{
#if PLATFORM(MAC)
    return [NSValue valueWithRect:rect];
#else
    return [NSValue valueWithCGRect:rect];
#endif
}

#if PLATFORM(MAC) && !defined(NSGEOMETRY_TYPES_SAME_AS_CGGEOMETRY_TYPES)

IntRect::operator NSRect() const
{
    return NSMakeRect(x(), y(), width(), height());
}

IntRect enclosingIntRect(const NSRect& rect)
{
    int left = clampTo<int>(std::floor(rect.origin.x));
    int top = clampTo<int>(std::floor(rect.origin.y));
    int right = clampTo<int>(std::ceil(NSMaxX(rect)));
    int bottom = clampTo<int>(std::ceil(NSMaxY(rect)));
    return { left, top, right - left, bottom - top };
}

#endif

}
