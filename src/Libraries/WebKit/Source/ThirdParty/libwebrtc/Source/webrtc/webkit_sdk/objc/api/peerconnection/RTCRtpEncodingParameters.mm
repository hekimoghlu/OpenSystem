/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
#import "RTCRtpEncodingParameters+Private.h"

#import "helpers/NSString+StdString.h"

@implementation RTCRtpEncodingParameters

@synthesize rid = _rid;
@synthesize isActive = _isActive;
@synthesize maxBitrateBps = _maxBitrateBps;
@synthesize minBitrateBps = _minBitrateBps;
@synthesize maxFramerate = _maxFramerate;
@synthesize numTemporalLayers = _numTemporalLayers;
@synthesize scaleResolutionDownBy = _scaleResolutionDownBy;
@synthesize ssrc = _ssrc;
@synthesize networkPriority = _networkPriority;

- (instancetype)init {
  return [super init];
}

- (instancetype)initWithNativeParameters:
    (const webrtc::RtpEncodingParameters &)nativeParameters {
  if (self = [self init]) {
    if (!nativeParameters.rid.empty()) {
      _rid = [NSString rtcStringForStdString:nativeParameters.rid];
    }
    _isActive = nativeParameters.active;
    if (nativeParameters.max_bitrate_bps) {
      _maxBitrateBps =
          [NSNumber numberWithInt:*nativeParameters.max_bitrate_bps];
    }
    if (nativeParameters.min_bitrate_bps) {
      _minBitrateBps =
          [NSNumber numberWithInt:*nativeParameters.min_bitrate_bps];
    }
    if (nativeParameters.max_framerate) {
      _maxFramerate = [NSNumber numberWithInt:*nativeParameters.max_framerate];
    }
    if (nativeParameters.num_temporal_layers) {
      _numTemporalLayers = [NSNumber numberWithInt:*nativeParameters.num_temporal_layers];
    }
    if (nativeParameters.scale_resolution_down_by) {
      _scaleResolutionDownBy =
          [NSNumber numberWithDouble:*nativeParameters.scale_resolution_down_by];
    }
    if (nativeParameters.ssrc) {
      _ssrc = [NSNumber numberWithUnsignedLong:*nativeParameters.ssrc];
    }
    _networkPriority =
        [RTCRtpEncodingParameters priorityFromNativePriority:nativeParameters.network_priority];
  }
  return self;
}

- (webrtc::RtpEncodingParameters)nativeParameters {
  webrtc::RtpEncodingParameters parameters;
  if (_rid != nil) {
    parameters.rid = [NSString rtcStdStringForString:_rid];
  }
  parameters.active = _isActive;
  if (_maxBitrateBps != nil) {
    parameters.max_bitrate_bps = std::optional<int>(_maxBitrateBps.intValue);
  }
  if (_minBitrateBps != nil) {
    parameters.min_bitrate_bps = std::optional<int>(_minBitrateBps.intValue);
  }
  if (_maxFramerate != nil) {
    parameters.max_framerate = std::optional<int>(_maxFramerate.intValue);
  }
  if (_numTemporalLayers != nil) {
    parameters.num_temporal_layers = std::optional<int>(_numTemporalLayers.intValue);
  }
  if (_scaleResolutionDownBy != nil) {
    parameters.scale_resolution_down_by =
      std::optional<double>(_scaleResolutionDownBy.doubleValue);
  }
  if (_ssrc != nil) {
    parameters.ssrc = std::optional<uint32_t>(_ssrc.unsignedLongValue);
  }
  parameters.network_priority =
      [RTCRtpEncodingParameters nativePriorityFromPriority:_networkPriority];
  return parameters;
}

+ (webrtc::Priority)nativePriorityFromPriority:(RTCPriority)networkPriority {
  switch (networkPriority) {
    case RTCPriorityVeryLow:
      return webrtc::Priority::kVeryLow;
    case RTCPriorityLow:
      return webrtc::Priority::kLow;
    case RTCPriorityMedium:
      return webrtc::Priority::kMedium;
    case RTCPriorityHigh:
      return webrtc::Priority::kHigh;
  }
}

+ (RTCPriority)priorityFromNativePriority:(webrtc::Priority)nativePriority {
  switch (nativePriority) {
    case webrtc::Priority::kVeryLow:
      return RTCPriorityVeryLow;
    case webrtc::Priority::kLow:
      return RTCPriorityLow;
    case webrtc::Priority::kMedium:
      return RTCPriorityMedium;
    case webrtc::Priority::kHigh:
      return RTCPriorityHigh;
  }
}

@end
