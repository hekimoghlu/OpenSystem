/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
#import "RTCRtpCodecCapability.h"

#include "api/rtp_parameters.h"

NS_ASSUME_NONNULL_BEGIN

@interface RTC_OBJC_TYPE (RTCRtpCodecCapability)()

/**
 * The native RtpCodecCapability representation of this RTCRtpCodecCapability
 * object. This is needed to pass to the underlying C++ APIs.
 */
@property(nonatomic, readonly) webrtc::RtpCodecCapability nativeRtpCodecCapability;

/**
 * Initialize an RTCRtpCodecCapability from a native RtpCodecCapability.
 */
- (instancetype)initWithNativeRtpCodecCapability:
    (const webrtc::RtpCodecCapability &)nativeRtpCodecCapability;

@end

NS_ASSUME_NONNULL_END
