/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#import "RTCVideoTrack.h"

#include "api/media_stream_interface.h"

NS_ASSUME_NONNULL_BEGIN

@interface RTCVideoTrack ()

/** VideoTrackInterface created or passed in at construction. */
@property(nonatomic, readonly) rtc::scoped_refptr<webrtc::VideoTrackInterface> nativeVideoTrack;

/** Initialize an RTCVideoTrack with its source and an id. */
- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                         source:(RTCVideoSource *)source
                        trackId:(NSString *)trackId;

@end

NS_ASSUME_NONNULL_END
