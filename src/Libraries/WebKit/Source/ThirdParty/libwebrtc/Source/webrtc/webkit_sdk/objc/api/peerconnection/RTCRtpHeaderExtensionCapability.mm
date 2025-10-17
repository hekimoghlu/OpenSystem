/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#import "RTCRtpHeaderExtensionCapability+Private.h"

#import "helpers/NSString+StdString.h"

@implementation RTC_OBJC_TYPE (RTCRtpHeaderExtensionCapability)

@synthesize uri = _uri;
@synthesize preferredId = _preferredId;
@synthesize preferredEncrypted = _preferredEncrypted;

- (instancetype)init {
  webrtc::RtpHeaderExtensionCapability nativeRtpHeaderExtensionCapability;
  return [self initWithNativeRtpHeaderExtensionCapability:nativeRtpHeaderExtensionCapability];
}

- (instancetype)initWithNativeRtpHeaderExtensionCapability:
    (const webrtc::RtpHeaderExtensionCapability &)nativeRtpHeaderExtensionCapability {
  if (self = [super init]) {
    _uri = [NSString stringForStdString:nativeRtpHeaderExtensionCapability.uri];
    if (nativeRtpHeaderExtensionCapability.preferred_id) {
      _preferredId = [NSNumber numberWithInt:*nativeRtpHeaderExtensionCapability.preferred_id];
    }
    _preferredEncrypted = nativeRtpHeaderExtensionCapability.preferred_encrypt;
  }
  return self;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"RTC_OBJC_TYPE(RTCRtpHeaderExtensionCapability) {\n  uri: "
                                    @"%@\n  preferredId: %@\n  preferredEncrypted: %d\n}",
                                    _uri,
                                    _preferredId,
                                    _preferredEncrypted];
}

- (webrtc::RtpHeaderExtensionCapability)nativeRtpHeaderExtensionCapability {
  webrtc::RtpHeaderExtensionCapability rtpHeaderExtensionCapability;
  rtpHeaderExtensionCapability.uri = [NSString stdStringForString:_uri];
  if (_preferredId != nil) {
    rtpHeaderExtensionCapability.preferred_id = absl::optional<int>(_preferredId.intValue);
  }
  rtpHeaderExtensionCapability.preferred_encrypt = _preferredEncrypted;
  return rtpHeaderExtensionCapability;
}

@end
