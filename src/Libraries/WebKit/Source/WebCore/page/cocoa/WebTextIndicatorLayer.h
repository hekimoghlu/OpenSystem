/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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

#import "TextIndicator.h"
#import <QuartzCore/CALayer.h>
#import <wtf/Noncopyable.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/RunLoop.h>

WEBCORE_EXPORT @interface WebTextIndicatorLayer : CALayer {
    RefPtr<WebCore::TextIndicator> _textIndicator;
    RetainPtr<NSArray> _bounceLayers;
    CGSize _margin;
    bool _hasCompletedAnimation;
    BOOL _fadingOut;
}

- (instancetype)initWithFrame:(CGRect)frame textIndicator:(WebCore::TextIndicator&)textIndicator margin:(CGSize)margin offset:(CGPoint)offset;

- (bool)indicatorWantsManualAnimation:(const WebCore::TextIndicator&)indicator;
- (bool)indicatorWantsBounce:(const WebCore::TextIndicator&)indicator;

- (void)present;
- (void)hideWithCompletionHandler:(void(^)(void))completionHandler;

- (void)setAnimationProgress:(float)progress;
- (BOOL)hasCompletedAnimation;

@property (nonatomic, getter=isFadingOut) BOOL fadingOut;

@end

