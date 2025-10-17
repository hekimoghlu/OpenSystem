/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#import <Cocoa/Cocoa.h>
#import <wtf/RetainPtr.h>

@interface WebWindowScaleAnimation : NSAnimation {
@private
    NSRect _initialFrame, _finalFrame, _realFrame;
    NSWindow *_window; // (assign)
    RetainPtr<NSAnimation> _subAnimation;
    NSTimeInterval _hintedDuration;
}
- (id)initWithHintedDuration:(NSTimeInterval)duration window:(NSWindow *)window initalFrame:(NSRect)initialFrame finalFrame:(NSRect)finalFrame;

- (void)setSubAnimation:(NSAnimation *)animation;

- (NSRect)currentFrame;

// Be sure to call setWindow:nil to clear the weak link _window when appropriate
- (void)setWindow:(NSWindow *)window;
@end


@interface WebWindowFadeAnimation : NSAnimation {
@private
    CGFloat _initialAlpha, _finalAlpha;
    NSWindow *_window; // (assign)
    BOOL _isStopped;
    
}
- (id)initWithDuration:(NSTimeInterval)duration window:(NSWindow *)window initialAlpha:(CGFloat)initialAlpha finalAlpha:(CGFloat)finalAlpha;

- (CGFloat)currentAlpha;

// Be sure to call setWindow:nil to clear the weak link _window when appropriate
- (void)setWindow:(NSWindow *)window;
@end
