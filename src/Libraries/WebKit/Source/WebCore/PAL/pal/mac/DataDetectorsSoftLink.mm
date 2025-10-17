/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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

#if PLATFORM(MAC) && ENABLE(DATA_DETECTION)

#import <AppKit/AppKit.h>
#import <pal/spi/mac/DataDetectorsSPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, PAL_EXPORT)

#if HAVE(SECURE_ACTION_CONTEXT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDSecureActionContext, PAL_EXPORT)
#else
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDActionContext, PAL_EXPORT)
#endif
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDActionsManager, PAL_EXPORT)

#if HAVE(DATA_DETECTORS_MAC_ACTION)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDMacAction, PAL_EXPORT)
#else
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDAction, PAL_EXPORT)
#endif

SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDHighlightCreateWithRectsInVisibleRectWithStyleScaleAndDirection, DDHighlightRef, (CFAllocatorRef allocator, CGRect* rects, CFIndex count, CGRect globalVisibleRect, DDHighlightStyle style, Boolean withButton, NSWritingDirection writingDirection, Boolean endsWithEOL, Boolean flipped, CGFloat scale), (allocator, rects, count, globalVisibleRect, style, withButton, writingDirection, endsWithEOL, flipped, scale), PAL_EXPORT)

SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDHighlightGetLayerWithContext, CGLayerRef, (DDHighlightRef highlight, CGContextRef context), (highlight, context), PAL_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDHighlightGetBoundingRect, CGRect, (DDHighlightRef highlight), (highlight), PAL_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, DataDetectors, DDHighlightPointIsOnHighlight, Boolean, (DDHighlightRef highlight, CGPoint point, Boolean* onButton), (highlight, point, onButton), PAL_EXPORT)

#endif // PLATFORM(MAC) && ENABLE(DATA_DETECTION)
