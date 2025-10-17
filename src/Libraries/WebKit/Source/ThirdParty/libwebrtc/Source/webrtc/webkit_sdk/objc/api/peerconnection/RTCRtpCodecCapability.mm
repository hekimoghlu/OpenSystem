/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
#import "RTCRtpCodecCapability+Private.h"

#import "RTCMediaStreamTrack.h"
#import "helpers/NSString+StdString.h"

#include "media/base/media_constants.h"
#include "rtc_base/checks.h"

@implementation RTC_OBJC_TYPE (RTCRtpCodecCapability)

@synthesize preferredPayloadType = _preferredPayloadType;
@synthesize name = _name;
@synthesize kind = _kind;
@synthesize clockRate = _clockRate;
@synthesize numChannels = _numChannels;
@synthesize parameters = _parameters;
@synthesize mimeType = _mimeType;

- (instancetype)init {
  webrtc::RtpCodecCapability rtpCodecCapability;
  return [self initWithNativeRtpCodecCapability:rtpCodecCapability];
}

- (instancetype)initWithNativeRtpCodecCapability:
    (const webrtc::RtpCodecCapability &)nativeRtpCodecCapability {
  if (self = [super init]) {
    if (nativeRtpCodecCapability.preferred_payload_type) {
      _preferredPayloadType =
          [NSNumber numberWithInt:*nativeRtpCodecCapability.preferred_payload_type];
    }
    _name = [NSString stringForStdString:nativeRtpCodecCapability.name];
    switch (nativeRtpCodecCapability.kind) {
      case cricket::MEDIA_TYPE_AUDIO:
        _kind = kRTCMediaStreamTrackKindAudio;
        break;
      case cricket::MEDIA_TYPE_VIDEO:
        _kind = kRTCMediaStreamTrackKindVideo;
        break;
      case cricket::MEDIA_TYPE_DATA:
        RTC_DCHECK_NOTREACHED();
        break;
      case cricket::MEDIA_TYPE_UNSUPPORTED:
        RTC_DCHECK_NOTREACHED();
        break;
    }
    if (nativeRtpCodecCapability.clock_rate) {
      _clockRate = [NSNumber numberWithInt:*nativeRtpCodecCapability.clock_rate];
    }
    if (nativeRtpCodecCapability.num_channels) {
      _numChannels = [NSNumber numberWithInt:*nativeRtpCodecCapability.num_channels];
    }
    NSMutableDictionary *parameters = [NSMutableDictionary dictionary];
    for (const auto &parameter : nativeRtpCodecCapability.parameters) {
      [parameters setObject:[NSString stringForStdString:parameter.second]
                     forKey:[NSString stringForStdString:parameter.first]];
    }
    _parameters = parameters;
    _mimeType = [NSString stringForStdString:nativeRtpCodecCapability.mime_type()];
  }
  return self;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"RTC_OBJC_TYPE(RTCRtpCodecCapability) {\n  "
                                    @"preferredPayloadType: %@\n  name: %@\n  kind: %@\n  "
                                    @"clockRate: %@\n  numChannels: %@\n  parameters: %@\n  "
                                    @"mimeType: %@\n}",
                                    _preferredPayloadType,
                                    _name,
                                    _kind,
                                    _clockRate,
                                    _numChannels,
                                    _parameters,
                                    _mimeType];
}

- (webrtc::RtpCodecCapability)nativeRtpCodecCapability {
  webrtc::RtpCodecCapability rtpCodecCapability;
  if (_preferredPayloadType != nil) {
    rtpCodecCapability.preferred_payload_type = absl::optional<int>(_preferredPayloadType.intValue);
  }
  rtpCodecCapability.name = [NSString stdStringForString:_name];
  // NSString pointer comparison is safe here since "kind" is readonly and only
  // populated above.
  if (_kind == kRTCMediaStreamTrackKindAudio) {
    rtpCodecCapability.kind = cricket::MEDIA_TYPE_AUDIO;
  } else if (_kind == kRTCMediaStreamTrackKindVideo) {
    rtpCodecCapability.kind = cricket::MEDIA_TYPE_VIDEO;
  } else {
    RTC_DCHECK_NOTREACHED();
  }
  if (_clockRate != nil) {
    rtpCodecCapability.clock_rate = absl::optional<int>(_clockRate.intValue);
  }
  if (_numChannels != nil) {
    rtpCodecCapability.num_channels = absl::optional<int>(_numChannels.intValue);
  }
  for (NSString *paramKey in _parameters.allKeys) {
    std::string key = [NSString stdStringForString:paramKey];
    std::string value = [NSString stdStringForString:_parameters[paramKey]];
    rtpCodecCapability.parameters[key] = value;
  }
  return rtpCodecCapability;
}

@end
