/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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

#import "RTCMacros.h"
#import "RTCVideoFrame.h"
#import "RTCVideoRenderer.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * RTCMTLVideoView is thin wrapper around MTKView.
 *
 * It has id<RTCVideoRenderer> property that renders video frames in the view's
 * bounds using Metal.
 * NOTE: always check if metal is available on the running device via
 * RTC_SUPPORTS_METAL macro before initializing this class.
 */
NS_CLASS_AVAILABLE_IOS(9)

RTC_OBJC_EXPORT
@interface RTCMTLVideoView : UIView<RTCVideoRenderer>

@property(nonatomic, weak) id<RTCVideoViewDelegate> delegate;

@property(nonatomic) UIViewContentMode videoContentMode;

/** @abstract Enables/disables rendering.
 */
@property(nonatomic, getter=isEnabled) BOOL enabled;

/** @abstract Wrapped RTCVideoRotation, or nil.
 */
@property(nonatomic, nullable) NSValue* rotationOverride;

@end

NS_ASSUME_NONNULL_END
