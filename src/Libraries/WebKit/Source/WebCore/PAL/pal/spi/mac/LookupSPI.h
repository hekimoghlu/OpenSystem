/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#if PLATFORM(MAC)

#import <objc/runtime.h>
#import <pal/mac/LookupSoftLink.h>
#import <pal/spi/mac/NSImmediateActionGestureRecognizerSPI.h>

#if USE(APPLE_INTERNAL_SDK)

#import <Lookup/Lookup.h>

#else

@interface LULookupDefinitionModule : NSObject

+ (NSRange)tokenRangeForString:(NSString *)string range:(NSRange)range options:(NSDictionary **)options;
+ (void)showDefinitionForTerm:(NSAttributedString *)term atLocation:(NSPoint)screenPoint options:(NSDictionary *)options;
+ (void)showDefinitionForTerm:(NSAttributedString *)term relativeToRect:(NSRect)positioningRect ofView:(NSView *)positioningView options:(NSDictionary *)options;
+ (void)hideDefinition;
+ (id<NSImmediateActionAnimationController>)lookupAnimationControllerForTerm:(NSAttributedString *)term atLocation:(NSPoint)screenPoint options:(NSDictionary *)options;
+ (id<NSImmediateActionAnimationController>)lookupAnimationControllerForTerm:(NSAttributedString *)term relativeToRect:(NSRect)positioningRect ofView:(NSView *)positioningView options:(NSDictionary *)options;

@end

#endif // !USE(APPLE_INTERNAL_SDK)

#endif // PLATFORM(MAC)
