/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
#import "RTCSessionDescription+Private.h"

#import "base/RTCLogging.h"
#import "helpers/NSString+StdString.h"

#include "rtc_base/checks.h"

@implementation RTCSessionDescription

@synthesize type = _type;
@synthesize sdp = _sdp;

+ (NSString *)stringForType:(RTCSdpType)type {
  std::string string = [[self class] stdStringForType:type];
  return [NSString stringForStdString:string];
}

+ (RTCSdpType)typeForString:(NSString *)string {
  std::string typeString = string.stdString;
  return [[self class] typeForStdString:typeString];
}

- (instancetype)initWithType:(RTCSdpType)type sdp:(NSString *)sdp {
  NSParameterAssert(sdp.length);
  if (self = [super init]) {
    _type = type;
    _sdp = [sdp copy];
  }
  return self;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"RTCSessionDescription:\n%@\n%@",
                                    [[self class] stringForType:_type],
                                    _sdp];
}

#pragma mark - Private

- (webrtc::SessionDescriptionInterface *)nativeDescription {
  webrtc::SdpParseError error;

  webrtc::SessionDescriptionInterface *description =
      webrtc::CreateSessionDescription([[self class] stdStringForType:_type],
                                       _sdp.stdString,
                                       &error);

  if (!description) {
    RTCLogError(@"Failed to create session description: %s\nline: %s",
                error.description.c_str(),
                error.line.c_str());
  }

  return description;
}

- (instancetype)initWithNativeDescription:
    (const webrtc::SessionDescriptionInterface *)nativeDescription {
  NSParameterAssert(nativeDescription);
  std::string sdp;
  nativeDescription->ToString(&sdp);
  RTCSdpType type = [[self class] typeForStdString:nativeDescription->type()];

  return [self initWithType:type
                        sdp:[NSString stringForStdString:sdp]];
}

+ (std::string)stdStringForType:(RTCSdpType)type {
  switch (type) {
    case RTCSdpTypeOffer:
      return webrtc::SessionDescriptionInterface::kOffer;
    case RTCSdpTypePrAnswer:
      return webrtc::SessionDescriptionInterface::kPrAnswer;
    case RTCSdpTypeAnswer:
      return webrtc::SessionDescriptionInterface::kAnswer;
  }
}

+ (RTCSdpType)typeForStdString:(const std::string &)string {
  if (string == webrtc::SessionDescriptionInterface::kOffer) {
    return RTCSdpTypeOffer;
  } else if (string == webrtc::SessionDescriptionInterface::kPrAnswer) {
    return RTCSdpTypePrAnswer;
  } else if (string == webrtc::SessionDescriptionInterface::kAnswer) {
    return RTCSdpTypeAnswer;
  } else {
    RTC_NOTREACHED();
    return RTCSdpTypeOffer;
  }
}

@end
