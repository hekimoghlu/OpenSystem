/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#import "RTCRtpParameters+Private.h"

#import "RTCRtcpParameters+Private.h"
#import "RTCRtpCodecParameters+Private.h"
#import "RTCRtpEncodingParameters+Private.h"
#import "RTCRtpHeaderExtension+Private.h"
#import "helpers/NSString+StdString.h"

@implementation RTCRtpParameters

@synthesize transactionId = _transactionId;
@synthesize rtcp = _rtcp;
@synthesize headerExtensions = _headerExtensions;
@synthesize encodings = _encodings;
@synthesize codecs = _codecs;
@synthesize degradationPreference = _degradationPreference;

- (instancetype)init {
  return [super init];
}

- (instancetype)initWithNativeParameters:
    (const webrtc::RtpParameters &)nativeParameters {
  if (self = [self init]) {
    _transactionId = [NSString stringForStdString:nativeParameters.transaction_id];
    _rtcp = [[RTCRtcpParameters alloc] initWithNativeParameters:nativeParameters.rtcp];

    NSMutableArray *headerExtensions = [[NSMutableArray alloc] init];
    for (const auto &headerExtension : nativeParameters.header_extensions) {
      [headerExtensions
          addObject:[[RTCRtpHeaderExtension alloc] initWithNativeParameters:headerExtension]];
    }
    _headerExtensions = headerExtensions;

    NSMutableArray *encodings = [[NSMutableArray alloc] init];
    for (const auto &encoding : nativeParameters.encodings) {
      [encodings addObject:[[RTCRtpEncodingParameters alloc]
                               initWithNativeParameters:encoding]];
    }
    _encodings = encodings;

    NSMutableArray *codecs = [[NSMutableArray alloc] init];
    for (const auto &codec : nativeParameters.codecs) {
      [codecs addObject:[[RTCRtpCodecParameters alloc]
                            initWithNativeParameters:codec]];
    }
    _codecs = codecs;

    _degradationPreference = [RTCRtpParameters
        degradationPreferenceFromNativeDegradationPreference:nativeParameters
                                                                 .degradation_preference];
  }
  return self;
}

- (webrtc::RtpParameters)nativeParameters {
  webrtc::RtpParameters parameters;
  parameters.transaction_id = [NSString stdStringForString:_transactionId];
  parameters.rtcp = [_rtcp nativeParameters];
  for (RTCRtpHeaderExtension *headerExtension in _headerExtensions) {
    parameters.header_extensions.push_back(headerExtension.nativeParameters);
  }
  for (RTCRtpEncodingParameters *encoding in _encodings) {
    parameters.encodings.push_back(encoding.nativeParameters);
  }
  for (RTCRtpCodecParameters *codec in _codecs) {
    parameters.codecs.push_back(codec.nativeParameters);
  }
  if (_degradationPreference) {
    parameters.degradation_preference = [RTCRtpParameters
        nativeDegradationPreferenceFromDegradationPreference:(RTCDegradationPreference)
                                                                 _degradationPreference.intValue];
  }
  return parameters;
}

+ (webrtc::DegradationPreference)nativeDegradationPreferenceFromDegradationPreference:
    (RTCDegradationPreference)degradationPreference {
  switch (degradationPreference) {
    case RTCDegradationPreferenceDisabled:
      return webrtc::DegradationPreference::DISABLED;
    case RTCDegradationPreferenceMaintainFramerate:
      return webrtc::DegradationPreference::MAINTAIN_FRAMERATE;
    case RTCDegradationPreferenceMaintainResolution:
      return webrtc::DegradationPreference::MAINTAIN_RESOLUTION;
    case RTCDegradationPreferenceBalanced:
      return webrtc::DegradationPreference::BALANCED;
  }
}

+ (NSNumber *)degradationPreferenceFromNativeDegradationPreference:
    (absl::optional<webrtc::DegradationPreference>)nativeDegradationPreference {
  if (!nativeDegradationPreference.has_value()) {
    return nil;
  }

  switch (*nativeDegradationPreference) {
    case webrtc::DegradationPreference::DISABLED:
      return @(RTCDegradationPreferenceDisabled);
    case webrtc::DegradationPreference::MAINTAIN_FRAMERATE:
      return @(RTCDegradationPreferenceMaintainFramerate);
    case webrtc::DegradationPreference::MAINTAIN_RESOLUTION:
      return @(RTCDegradationPreferenceMaintainResolution);
    case webrtc::DegradationPreference::BALANCED:
      return @(RTCDegradationPreferenceBalanced);
  }
}

@end
