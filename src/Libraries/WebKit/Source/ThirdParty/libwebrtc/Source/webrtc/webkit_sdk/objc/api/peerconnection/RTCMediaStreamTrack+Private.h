/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
#import "RTCMediaStreamTrack.h"

#include "api/media_stream_interface.h"

typedef NS_ENUM(NSInteger, RTCMediaStreamTrackType) {
  RTCMediaStreamTrackTypeAudio,
  RTCMediaStreamTrackTypeVideo,
};

NS_ASSUME_NONNULL_BEGIN

@class RTCPeerConnectionFactory;

@interface RTCMediaStreamTrack ()

@property(nonatomic, readonly) RTCPeerConnectionFactory *factory;

/**
 * The native MediaStreamTrackInterface passed in or created during
 * construction.
 */
@property(nonatomic, readonly) rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> nativeTrack;

/**
 * Initialize an RTCMediaStreamTrack from a native MediaStreamTrackInterface.
 */
- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                    nativeTrack:(rtc::scoped_refptr<webrtc::MediaStreamTrackInterface>)nativeTrack
                           type:(RTCMediaStreamTrackType)type NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                    nativeTrack:(rtc::scoped_refptr<webrtc::MediaStreamTrackInterface>)nativeTrack;

- (BOOL)isEqualToTrack:(RTCMediaStreamTrack *)track;

+ (webrtc::MediaStreamTrackInterface::TrackState)nativeTrackStateForState:
        (RTCMediaStreamTrackState)state;

+ (RTCMediaStreamTrackState)trackStateForNativeState:
        (webrtc::MediaStreamTrackInterface::TrackState)nativeState;

+ (NSString *)stringForState:(RTCMediaStreamTrackState)state;

+ (RTCMediaStreamTrack *)mediaTrackForNativeTrack:
                             (rtc::scoped_refptr<webrtc::MediaStreamTrackInterface>)nativeTrack
                                          factory:(RTCPeerConnectionFactory *)factory;

@end

NS_ASSUME_NONNULL_END
