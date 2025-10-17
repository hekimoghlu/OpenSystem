/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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

/**
 * Represents the state of the track. This exposes the same states in C++.
 */
typedef NS_ENUM(NSInteger, RTCMediaStreamTrackState) {
  RTCMediaStreamTrackStateLive,
  RTCMediaStreamTrackStateEnded
};

NS_ASSUME_NONNULL_BEGIN

RTC_EXTERN NSString *const kRTCMediaStreamTrackKindAudio;
RTC_EXTERN NSString *const kRTCMediaStreamTrackKindVideo;

RTC_OBJC_EXPORT
@interface RTCMediaStreamTrack : NSObject

/**
 * The kind of track. For example, "audio" if this track represents an audio
 * track and "video" if this track represents a video track.
 */
@property(nonatomic, readonly) NSString *kind;

/** An identifier string. */
@property(nonatomic, readonly) NSString *trackId;

/** The enabled state of the track. */
@property(nonatomic, assign) BOOL isEnabled;

/** The state of the track. */
@property(nonatomic, readonly) RTCMediaStreamTrackState readyState;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
