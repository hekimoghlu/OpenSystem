/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#import "RTCRtpCodecParameters+Private.h"

#import "RTCMediaStreamTrack.h"
#import "helpers/NSString+StdString.h"

#include "media/base/media_constants.h"
#include "rtc_base/checks.h"

const NSString * const kRTCRtxCodecName = @(cricket::kRtxCodecName);
const NSString * const kRTCRedCodecName = @(cricket::kRedCodecName);
const NSString * const kRTCUlpfecCodecName = @(cricket::kUlpfecCodecName);
const NSString * const kRTCFlexfecCodecName = @(cricket::kFlexfecCodecName);
const NSString * const kRTCOpusCodecName = @(cricket::kOpusCodecName);
const NSString * const kRTCIsacCodecName = @(cricket::kIsacCodecName);
const NSString * const kRTCL16CodecName  = @(cricket::kL16CodecName);
const NSString * const kRTCG722CodecName = @(cricket::kG722CodecName);
const NSString * const kRTCIlbcCodecName = @(cricket::kIlbcCodecName);
const NSString * const kRTCPcmuCodecName = @(cricket::kPcmuCodecName);
const NSString * const kRTCPcmaCodecName = @(cricket::kPcmaCodecName);
const NSString * const kRTCDtmfCodecName = @(cricket::kDtmfCodecName);
const NSString * const kRTCComfortNoiseCodecName =
    @(cricket::kComfortNoiseCodecName);
const NSString * const kRTCVp8CodecName = @(cricket::kVp8CodecName);
const NSString * const kRTCVp9CodecName = @(cricket::kVp9CodecName);
const NSString * const kRTCH264CodecName = @(cricket::kH264CodecName);

@implementation RTCRtpCodecParameters

@synthesize payloadType = _payloadType;
@synthesize name = _name;
@synthesize kind = _kind;
@synthesize clockRate = _clockRate;
@synthesize numChannels = _numChannels;
@synthesize parameters = _parameters;

- (instancetype)init {
  return [super init];
}

- (instancetype)initWithNativeParameters:
    (const webrtc::RtpCodecParameters &)nativeParameters {
  if (self = [self init]) {
    _payloadType = nativeParameters.payload_type;
    _name = [NSString stringForStdString:nativeParameters.name];
    switch (nativeParameters.kind) {
      case cricket::MEDIA_TYPE_AUDIO:
        _kind = kRTCMediaStreamTrackKindAudio;
        break;
      case cricket::MEDIA_TYPE_VIDEO:
        _kind = kRTCMediaStreamTrackKindVideo;
        break;
      case cricket::MEDIA_TYPE_DATA:
        RTC_NOTREACHED();
        break;
    }
    if (nativeParameters.clock_rate) {
      _clockRate = [NSNumber numberWithInt:*nativeParameters.clock_rate];
    }
    if (nativeParameters.num_channels) {
      _numChannels = [NSNumber numberWithInt:*nativeParameters.num_channels];
    }
    NSMutableDictionary *parameters = [NSMutableDictionary dictionary];
    for (const auto &parameter : nativeParameters.parameters) {
      [parameters setObject:[NSString stringForStdString:parameter.second]
                     forKey:[NSString stringForStdString:parameter.first]];
    }
    _parameters = parameters;
  }
  return self;
}

- (webrtc::RtpCodecParameters)nativeParameters {
  webrtc::RtpCodecParameters parameters;
  parameters.payload_type = _payloadType;
  parameters.name = [NSString stdStringForString:_name];
  // NSString pointer comparison is safe here since "kind" is readonly and only
  // populated above.
  if (_kind == kRTCMediaStreamTrackKindAudio) {
    parameters.kind = cricket::MEDIA_TYPE_AUDIO;
  } else if (_kind == kRTCMediaStreamTrackKindVideo) {
    parameters.kind = cricket::MEDIA_TYPE_VIDEO;
  } else {
    RTC_NOTREACHED();
  }
  if (_clockRate != nil) {
    parameters.clock_rate = absl::optional<int>(_clockRate.intValue);
  }
  if (_numChannels != nil) {
    parameters.num_channels = absl::optional<int>(_numChannels.intValue);
  }
  for (NSString *paramKey in _parameters.allKeys) {
    std::string key = [NSString stdStringForString:paramKey];
    std::string value = [NSString stdStringForString:_parameters[paramKey]];
    parameters.parameters[key] = value;
  }
  return parameters;
}

@end
