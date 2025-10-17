/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#import "RTCVideoRendererAdapter.h"

#import "base/RTCVideoRenderer.h"

#include "api/media_stream_interface.h"

NS_ASSUME_NONNULL_BEGIN

@interface RTCVideoRendererAdapter ()

/**
 * The Objective-C video renderer passed to this adapter during construction.
 * Calls made to the webrtc::VideoRenderInterface will be adapted and passed to
 * this video renderer.
 */
@property(nonatomic, readonly) id<RTCVideoRenderer> videoRenderer;

/**
 * The native VideoSinkInterface surface exposed by this adapter. Calls made
 * to this interface will be adapted and passed to the RTCVideoRenderer supplied
 * during construction. This pointer is unsafe and owned by this class.
 */
@property(nonatomic, readonly) rtc::VideoSinkInterface<webrtc::VideoFrame> *nativeVideoRenderer;

/** Initialize an RTCVideoRendererAdapter with an RTCVideoRenderer. */
- (instancetype)initWithNativeRenderer:(id<RTCVideoRenderer>)videoRenderer
    NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
