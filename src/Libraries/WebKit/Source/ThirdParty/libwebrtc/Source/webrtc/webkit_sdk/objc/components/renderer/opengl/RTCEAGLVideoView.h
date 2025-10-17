/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#import "RTCMacros.h"
#import "RTCVideoRenderer.h"
#import "RTCVideoViewShading.h"

NS_ASSUME_NONNULL_BEGIN

@class RTCEAGLVideoView;

/**
 * RTCEAGLVideoView is an RTCVideoRenderer which renders video frames in its
 * bounds using OpenGLES 2.0 or OpenGLES 3.0.
 */
RTC_OBJC_EXPORT
NS_EXTENSION_UNAVAILABLE_IOS("Rendering not available in app extensions.")
@interface RTCEAGLVideoView : UIView <RTCVideoRenderer>

@property(nonatomic, weak) id<RTCVideoViewDelegate> delegate;

- (instancetype)initWithFrame:(CGRect)frame
                       shader:(id<RTCVideoViewShading>)shader NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithCoder:(NSCoder *)aDecoder
                       shader:(id<RTCVideoViewShading>)shader NS_DESIGNATED_INITIALIZER;

/** @abstract Wrapped RTCVideoRotation, or nil.
 */
@property(nonatomic, nullable) NSValue *rotationOverride;
@end

NS_ASSUME_NONNULL_END
