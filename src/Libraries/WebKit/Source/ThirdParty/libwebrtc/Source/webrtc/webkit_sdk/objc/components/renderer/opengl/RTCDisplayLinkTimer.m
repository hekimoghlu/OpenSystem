/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#import "RTCDisplayLinkTimer.h"

#import <UIKit/UIKit.h>

@implementation RTCDisplayLinkTimer {
  CADisplayLink *_displayLink;
  void (^_timerHandler)(void);
}

- (instancetype)initWithTimerHandler:(void (^)(void))timerHandler {
  NSParameterAssert(timerHandler);
  if (self = [super init]) {
    _timerHandler = timerHandler;
    _displayLink =
        [CADisplayLink displayLinkWithTarget:self
                                    selector:@selector(displayLinkDidFire:)];
    _displayLink.paused = YES;
#if __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_10_0
    _displayLink.preferredFramesPerSecond = 30;
#else
    [_displayLink setFrameInterval:2];
#endif
    [_displayLink addToRunLoop:[NSRunLoop currentRunLoop]
                       forMode:NSRunLoopCommonModes];
  }
  return self;
}

- (void)dealloc {
  [self invalidate];
}

- (BOOL)isPaused {
  return _displayLink.paused;
}

- (void)setIsPaused:(BOOL)isPaused {
  _displayLink.paused = isPaused;
}

- (void)invalidate {
  [_displayLink invalidate];
}

- (void)displayLinkDidFire:(CADisplayLink *)displayLink {
  _timerHandler();
}

@end
