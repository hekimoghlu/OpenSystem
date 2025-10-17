/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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

RTC_OBJC_EXPORT
@interface RTC_OBJC_TYPE (RTCRtpCodecCapability) : NSObject

/** The preferred RTP payload type. */
@property(nonatomic, readonly, nullable) NSNumber *preferredPayloadType;

/**
 * The codec MIME subtype. Valid types are listed in:
 * http://www.iana.org/assignments/rtp-parameters/rtp-parameters.xhtml#rtp-parameters-2
 *
 * Several supported types are represented by the constants above.
 */
@property(nonatomic, readonly) NSString *name;

/**
 * The media type of this codec. Equivalent to MIME top-level type.
 *
 * Valid values are kRTCMediaStreamTrackKindAudio and
 * kRTCMediaStreamTrackKindVideo.
 */
@property(nonatomic, readonly) NSString *kind;

/** The codec clock rate expressed in Hertz. */
@property(nonatomic, readonly, nullable) NSNumber *clockRate;

/**
 * The number of audio channels (mono=1, stereo=2).
 * Set to null for video codecs.
 **/
@property(nonatomic, readonly, nullable) NSNumber *numChannels;

/** The "format specific parameters" field from the "a=fmtp" line in the SDP */
@property(nonatomic, readonly) NSDictionary<NSString *, NSString *> *parameters;

/** The MIME type of the codec. */
@property(nonatomic, readonly) NSString *mimeType;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
