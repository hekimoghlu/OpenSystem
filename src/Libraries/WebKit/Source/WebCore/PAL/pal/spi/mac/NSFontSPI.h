/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
#import <wtf/Platform.h>

#if PLATFORM(MAC)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSFontDescriptor_Private.h>
#import <AppKit/NSFont_Private.h>

#else

@interface NSFont ()
+ (NSFont *)systemFontOfSize:(CGFloat)size weight:(CGFloat)weight;
@end

extern const CGFloat NSFontWeightUltraLight;
extern const CGFloat NSFontWeightThin;
extern const CGFloat NSFontWeightLight;
extern const CGFloat NSFontWeightRegular;
extern const CGFloat NSFontWeightMedium;
extern const CGFloat NSFontWeightSemibold;
extern const CGFloat NSFontWeightBold;
extern const CGFloat NSFontWeightHeavy;
extern const CGFloat NSFontWeightBlack;

#endif

#endif
