/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#import "FloatRect.h"

#if !PLATFORM(MAC)
#import <UIKit/UIGeometry.h>
#endif

namespace WebCore {

id makeNSArrayElement(const WebCore::FloatRect& rect)
{
#if PLATFORM(MAC)
    return [NSValue valueWithRect:rect];
#else
    return [NSValue valueWithCGRect:rect];
#endif
}

#if PLATFORM(MAC) && !defined(NSGEOMETRY_TYPES_SAME_AS_CGGEOMETRY_TYPES)

FloatRect::FloatRect(const NSRect& rect)
    : m_location(rect.origin)
    , m_size(rect.size)
{
}

FloatRect::operator NSRect() const
{
    return NSMakeRect(x(), y(), width(), height());
}

#endif

}
