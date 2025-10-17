/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSScrollingMomentumCalculator_Private.h>

#else

@interface _NSScrollingMomentumCalculator : NSObject

- (instancetype)initWithInitialOrigin:(NSPoint)origin velocity:(NSPoint)velocity documentFrame:(NSRect)docFrame constrainedClippingOrigin:(NSPoint)constrainedClippingOrigin clippingSize:(NSSize)clipViewSize tolerance:(NSSize)tolerance;
- (NSPoint)positionAfterDuration:(NSTimeInterval)duration;
- (void)calculateToReachDestination;

@property NSPoint destinationOrigin;
@property (readonly) NSTimeInterval durationUntilStop;

@end

#endif /* USE(APPLE_INTERNAL_SDK) */
