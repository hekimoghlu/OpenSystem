/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#if PLATFORM(IOS_FAMILY)

#import "WKScrollViewTrackingTapGestureRecognizer.h"

@class WKTouchEventsGestureRecognizer;

// The purpose of this class is to call a target/action when
// the gesture is recognized, as well as the typical time when
// a gesture should be handled. This allows it to be used while
// it is waiting for another gesture recognizer to fail.
@interface WKSyntheticTapGestureRecognizer : WKScrollViewTrackingTapGestureRecognizer
- (void)setGestureIdentifiedTarget:(id)target action:(SEL)action;
- (void)setGestureFailedTarget:(id)target action:(SEL)action;
- (void)setResetTarget:(id)target action:(SEL)action;
@property (nonatomic, weak) WKTouchEventsGestureRecognizer *supportingTouchEventsGestureRecognizer;
@property (nonatomic, readonly) NSNumber *lastActiveTouchIdentifier;
@end

#endif
