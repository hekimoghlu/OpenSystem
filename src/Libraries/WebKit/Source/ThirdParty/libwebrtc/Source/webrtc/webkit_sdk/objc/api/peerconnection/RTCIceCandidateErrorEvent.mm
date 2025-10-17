/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#import "RTCIceCandidateErrorEvent+Private.h"

#import "helpers/NSString+StdString.h"

@implementation RTC_OBJC_TYPE (RTCIceCandidateErrorEvent)

@synthesize address = _address;
@synthesize port = _port;
@synthesize url = _url;
@synthesize errorCode = _errorCode;
@synthesize errorText = _errorText;

- (instancetype)init {
  return [super init];
}

- (instancetype)initWithAddress:(const std::string&)address
                           port:(const int)port
                            url:(const std::string&)url
                      errorCode:(const int)errorCode
                      errorText:(const std::string&)errorText {
  if (self = [self init]) {
    _address = [NSString stringForStdString:address];
    _port = port;
    _url = [NSString stringForStdString:url];
    _errorCode = errorCode;
    _errorText = [NSString stringForStdString:errorText];
  }
  return self;
}

@end
