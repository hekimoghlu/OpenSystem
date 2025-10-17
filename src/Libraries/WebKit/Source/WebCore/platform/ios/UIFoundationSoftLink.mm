/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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

#if PLATFORM(IOS_FAMILY)

#import <pal/spi/ios/UIKitSPI.h>
#import <wtf/SoftLinking.h>

// FIXME: Remove SOFT_LINK_PRIVATE_FRAMEWORK(UIFoundation) and move symbols from NSAttributedStringSPI.h to here.
SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE(WebCore, UIFoundation)

SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSColor)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSTextAttachment)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSMutableParagraphStyle)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSTextList)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSTextBlock)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSTextTableBlock)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSTextTable)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSTextTab)

#if ENABLE(MULTI_REPRESENTATION_HEIC)

SOFT_LINK_CLASS_FOR_SOURCE(WebCore, UIFoundation, NSAdaptiveImageGlyph)

SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, UIFoundation, NSAdaptiveImageGlyphAttributeName, NSString *)

#endif

#endif
