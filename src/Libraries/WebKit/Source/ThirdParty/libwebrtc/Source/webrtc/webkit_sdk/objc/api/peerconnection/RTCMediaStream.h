/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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

NS_ASSUME_NONNULL_BEGIN

@class RTCAudioTrack;
@class RTCPeerConnectionFactory;
@class RTCVideoTrack;

RTC_OBJC_EXPORT
@interface RTCMediaStream : NSObject

/** The audio tracks in this stream. */
@property(nonatomic, strong, readonly) NSArray<RTCAudioTrack *> *audioTracks;

/** The video tracks in this stream. */
@property(nonatomic, strong, readonly) NSArray<RTCVideoTrack *> *videoTracks;

/** An identifier for this media stream. */
@property(nonatomic, readonly) NSString *streamId;

- (instancetype)init NS_UNAVAILABLE;

/** Adds the given audio track to this media stream. */
- (void)addAudioTrack:(RTCAudioTrack *)audioTrack;

/** Adds the given video track to this media stream. */
- (void)addVideoTrack:(RTCVideoTrack *)videoTrack;

/** Removes the given audio track to this media stream. */
- (void)removeAudioTrack:(RTCAudioTrack *)audioTrack;

/** Removes the given video track to this media stream. */
- (void)removeVideoTrack:(RTCVideoTrack *)videoTrack;

@end

NS_ASSUME_NONNULL_END
