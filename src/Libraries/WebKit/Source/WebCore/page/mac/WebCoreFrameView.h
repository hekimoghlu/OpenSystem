/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#import "ScrollTypes.h"
#import <wtf/Platform.h>
#import <wtf/NakedPtr.h>

#if PLATFORM(IOS_FAMILY)
#import "WAKAppKitStubs.h"
#endif

namespace WebCore {
class LocalFrame;
}

@protocol WebCoreFrameScrollView
- (void)setScrollingModes:(WebCore::ScrollbarMode)hMode vertical:(WebCore::ScrollbarMode)vMode andLock:(BOOL)lock;
- (void)scrollingModes:(WebCore::ScrollbarMode*)hMode vertical:(WebCore::ScrollbarMode*)vMode;

- (void)setScrollBarsSuppressed:(BOOL)suppressed repaintOnUnsuppress:(BOOL)repaint;
- (void)setScrollOrigin:(NSPoint)origin updatePositionAtAll:(BOOL)updatePositionAtAll immediately:(BOOL)updatePositionImmediately;
- (NSPoint)scrollOrigin;
@end

@protocol WebCoreFrameView
- (NakedPtr<WebCore::LocalFrame>)_web_frame;
@end
