/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
#import "RTCRtcpParameters.h"
#import "RTCRtpCodecParameters.h"
#import "RTCRtpEncodingParameters.h"
#import "RTCRtpHeaderExtension.h"

NS_ASSUME_NONNULL_BEGIN

/** Corresponds to webrtc::DegradationPreference. */
typedef NS_ENUM(NSInteger, RTCDegradationPreference) {
  RTCDegradationPreferenceDisabled,
  RTCDegradationPreferenceMaintainFramerate,
  RTCDegradationPreferenceMaintainResolution,
  RTCDegradationPreferenceBalanced
};

RTC_OBJC_EXPORT
@interface RTCRtpParameters : NSObject

/** A unique identifier for the last set of parameters applied. */
@property(nonatomic, copy) NSString *transactionId;

/** Parameters used for RTCP. */
@property(nonatomic, readonly, copy) RTCRtcpParameters *rtcp;

/** An array containing parameters for RTP header extensions. */
@property(nonatomic, readonly, copy) NSArray<RTCRtpHeaderExtension *> *headerExtensions;

/** The currently active encodings in the order of preference. */
@property(nonatomic, copy) NSArray<RTCRtpEncodingParameters *> *encodings;

/** The negotiated set of send codecs in order of preference. */
@property(nonatomic, copy) NSArray<RTCRtpCodecParameters *> *codecs;

/**
 * Degradation preference in case of CPU adaptation or constrained bandwidth.
 * If nil, implementation default degradation preference will be used.
 */
@property(nonatomic, copy, nullable) NSNumber *degradationPreference;

- (instancetype)init NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
