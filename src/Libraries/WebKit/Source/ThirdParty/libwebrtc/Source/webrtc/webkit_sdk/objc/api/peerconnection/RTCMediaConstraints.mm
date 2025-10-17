/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
#import "RTCMediaConstraints+Private.h"

#import "helpers/NSString+StdString.h"

#include <memory>

NSString *const kRTCMediaConstraintsAudioNetworkAdaptorConfig =
    @(webrtc::MediaConstraints::kAudioNetworkAdaptorConfig);

NSString *const kRTCMediaConstraintsIceRestart = @(webrtc::MediaConstraints::kIceRestart);
NSString *const kRTCMediaConstraintsOfferToReceiveAudio =
    @(webrtc::MediaConstraints::kOfferToReceiveAudio);
NSString *const kRTCMediaConstraintsOfferToReceiveVideo =
    @(webrtc::MediaConstraints::kOfferToReceiveVideo);
NSString *const kRTCMediaConstraintsVoiceActivityDetection =
    @(webrtc::MediaConstraints::kVoiceActivityDetection);

NSString *const kRTCMediaConstraintsValueTrue = @(webrtc::MediaConstraints::kValueTrue);
NSString *const kRTCMediaConstraintsValueFalse = @(webrtc::MediaConstraints::kValueFalse);

@implementation RTCMediaConstraints {
  NSDictionary<NSString *, NSString *> *_mandatory;
  NSDictionary<NSString *, NSString *> *_optional;
}

- (instancetype)initWithMandatoryConstraints:
    (NSDictionary<NSString *, NSString *> *)mandatory
                         optionalConstraints:
    (NSDictionary<NSString *, NSString *> *)optional {
  if (self = [super init]) {
    _mandatory = [[NSDictionary alloc] initWithDictionary:mandatory
                                                copyItems:YES];
    _optional = [[NSDictionary alloc] initWithDictionary:optional
                                               copyItems:YES];
  }
  return self;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"RTCMediaConstraints:\n%@\n%@",
                                    _mandatory,
                                    _optional];
}

#pragma mark - Private

- (std::unique_ptr<webrtc::MediaConstraints>)nativeConstraints {
  webrtc::MediaConstraints::Constraints mandatory =
      [[self class] nativeConstraintsForConstraints:_mandatory];
  webrtc::MediaConstraints::Constraints optional =
      [[self class] nativeConstraintsForConstraints:_optional];

  webrtc::MediaConstraints *nativeConstraints =
      new webrtc::MediaConstraints(mandatory, optional);
  return std::unique_ptr<webrtc::MediaConstraints>(nativeConstraints);
}

+ (webrtc::MediaConstraints::Constraints)nativeConstraintsForConstraints:
    (NSDictionary<NSString *, NSString *> *)constraints {
  webrtc::MediaConstraints::Constraints nativeConstraints;
  for (NSString *key in constraints) {
    NSAssert([key isKindOfClass:[NSString class]],
             @"%@ is not an NSString.", key);
    NSString *value = [constraints objectForKey:key];
    NSAssert([value isKindOfClass:[NSString class]],
             @"%@ is not an NSString.", value);
    if ([kRTCMediaConstraintsAudioNetworkAdaptorConfig isEqualToString:key]) {
      // This value is base64 encoded.
      NSData *charData = [[NSData alloc] initWithBase64EncodedString:value options:0];
      std::string configValue =
          std::string(reinterpret_cast<const char *>(charData.bytes), charData.length);
      nativeConstraints.push_back(webrtc::MediaConstraints::Constraint(key.stdString, configValue));
    } else {
      nativeConstraints.push_back(
          webrtc::MediaConstraints::Constraint(key.stdString, value.stdString));
    }
  }
  return nativeConstraints;
}

@end
