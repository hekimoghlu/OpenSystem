/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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

#if PLATFORM(MAC) && ENABLE(DATA_DETECTION)

#import <pal/spi/mac/DataDetectorsSPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, DataDetectors)

#if HAVE(SECURE_ACTION_CONTEXT)
SOFT_LINK_CLASS_FOR_HEADER(PAL, DDSecureActionContext)
#else
SOFT_LINK_CLASS_FOR_HEADER(PAL, DDActionContext)
#endif

SOFT_LINK_CLASS_FOR_HEADER(PAL, DDActionsManager)

#if HAVE(DATA_DETECTORS_MAC_ACTION)
SOFT_LINK_CLASS_FOR_HEADER(PAL, DDMacAction)
#else
SOFT_LINK_CLASS_FOR_HEADER(PAL, DDAction)
#endif

SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectors, DDHighlightCreateWithRectsInVisibleRectWithStyleScaleAndDirection, DDHighlightRef, (CFAllocatorRef allocator, CGRect* rects, CFIndex count, CGRect globalVisibleRect, DDHighlightStyle style, Boolean withButton, NSWritingDirection writingDirection, Boolean endsWithEOL, Boolean flipped, CGFloat scale), (allocator, rects, count, globalVisibleRect, style, withButton, writingDirection, endsWithEOL, flipped, scale))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectors, DDHighlightGetLayerWithContext, CGLayerRef, (DDHighlightRef highlight, CGContextRef context), (highlight, context))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectors, DDHighlightGetBoundingRect, CGRect, (DDHighlightRef highlight), (highlight))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectors, DDHighlightPointIsOnHighlight, Boolean, (DDHighlightRef highlight, CGPoint point, Boolean* onButton), (highlight, point, onButton))

namespace PAL {

inline WKDDActionContext *allocWKDDActionContextInstance()
{
#if HAVE(SECURE_ACTION_CONTEXT)
    return allocDDSecureActionContextInstance();
#else
    return allocDDActionContextInstance();
#endif
}

#ifdef __OBJC__
inline Class getWKDDActionContextClass()
{
#if HAVE(SECURE_ACTION_CONTEXT)
    return getDDSecureActionContextClass();
#else
    return getDDActionContextClass();
#endif
}
#endif

}

#endif // PLATFORM(MAC) && ENABLE(DATA_DETECTION)
