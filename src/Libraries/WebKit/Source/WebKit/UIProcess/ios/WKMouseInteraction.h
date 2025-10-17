/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

#if HAVE(UIKIT_WITH_MOUSE_SUPPORT)

#import "NativeWebMouseEvent.h"
#import <UIKit/UIKit.h>

@class WKMouseInteraction;

@protocol WKMouseInteractionDelegate<NSObject>
- (void)mouseInteraction:(WKMouseInteraction *)interaction changedWithEvent:(const WebKit::NativeWebMouseEvent&)event;
@end

@interface WKMouseInteraction : NSObject<UIInteraction>

- (instancetype)initWithDelegate:(id <WKMouseInteractionDelegate>)delegate;

- (CGPoint)locationInView:(UIView *)view;
- (BOOL)hasGesture:(UIGestureRecognizer *)gesture;

@property (nonatomic, getter=isEnabled) BOOL enabled;
@property (nonatomic, readonly, weak) id <WKMouseInteractionDelegate> delegate;
@property (nonatomic, readonly) UITouch *mouseTouch;
@property (nonatomic, readonly) std::optional<CGPoint> lastLocation;
@property (nonatomic, weak, readonly) UIView *view;
@property (nonatomic, readonly) UIGestureRecognizer *mouseTouchGestureRecognizer;

@end

#endif // HAVE(UIKIT_WITH_MOUSE_SUPPORT)
