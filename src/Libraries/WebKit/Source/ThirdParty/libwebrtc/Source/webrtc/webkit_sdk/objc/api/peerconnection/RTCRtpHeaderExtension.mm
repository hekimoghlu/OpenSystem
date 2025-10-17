/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#import "RTCRtpHeaderExtension+Private.h"

#import "helpers/NSString+StdString.h"

@implementation RTCRtpHeaderExtension

@synthesize uri = _uri;
@synthesize id = _id;
@synthesize encrypted = _encrypted;

- (instancetype)init {
  return [super init];
}

- (instancetype)initWithNativeParameters:(const webrtc::RtpExtension &)nativeParameters {
  if (self = [self init]) {
    _uri = [NSString stringForStdString:nativeParameters.uri];
    _id = nativeParameters.id;
    _encrypted = nativeParameters.encrypt;
  }
  return self;
}

- (webrtc::RtpExtension)nativeParameters {
  webrtc::RtpExtension extension;
  extension.uri = [NSString stdStringForString:_uri];
  extension.id = _id;
  extension.encrypt = _encrypted;
  return extension;
}

@end
