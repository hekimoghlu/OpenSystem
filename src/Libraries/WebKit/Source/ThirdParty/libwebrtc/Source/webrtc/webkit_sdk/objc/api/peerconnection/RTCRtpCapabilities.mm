/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#import "RTCRtpCapabilities+Private.h"

#import "RTCRtpCodecCapability+Private.h"
#import "RTCRtpHeaderExtensionCapability+Private.h"

#import "base/RTCLogging.h"
#import "helpers/NSString+StdString.h"

@implementation RTC_OBJC_TYPE (RTCRtpCapabilities)

@synthesize codecs = _codecs;
@synthesize headerExtensions = _headerExtensions;

- (instancetype)init {
  webrtc::RtpCapabilities nativeRtpCapabilities;
  return [self initWithNativeRtpCapabilities:nativeRtpCapabilities];
}

- (instancetype)initWithNativeRtpCapabilities:
    (const webrtc::RtpCapabilities &)nativeRtpCapabilities {
  if (self = [super init]) {
    NSMutableArray *codecs = [[NSMutableArray alloc] init];
    for (const auto &codec : nativeRtpCapabilities.codecs) {
      [codecs addObject:[[RTC_OBJC_TYPE(RTCRtpCodecCapability) alloc]
                            initWithNativeRtpCodecCapability:codec]];
    }
    _codecs = codecs;

    NSMutableArray *headerExtensions = [[NSMutableArray alloc] init];
    for (const auto &headerExtension : nativeRtpCapabilities.header_extensions) {
      [headerExtensions addObject:[[RTC_OBJC_TYPE(RTCRtpHeaderExtensionCapability) alloc]
                                      initWithNativeRtpHeaderExtensionCapability:headerExtension]];
    }
    _headerExtensions = headerExtensions;
  }
  return self;
}

- (webrtc::RtpCapabilities)nativeRtpCapabilities {
  webrtc::RtpCapabilities rtpCapabilities;
  for (RTC_OBJC_TYPE(RTCRtpCodecCapability) * codec in _codecs) {
    rtpCapabilities.codecs.push_back(codec.nativeRtpCodecCapability);
  }
  for (RTC_OBJC_TYPE(RTCRtpHeaderExtensionCapability) * headerExtension in _headerExtensions) {
    rtpCapabilities.header_extensions.push_back(headerExtension.nativeRtpHeaderExtensionCapability);
  }
  return rtpCapabilities;
}

@end
