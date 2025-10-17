/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#import <Foundation/Foundation.h>

#include <memory>

#include "rtc_base/gunit.h"

#import "api/peerconnection/RTCMediaConstraints+Private.h"
#import "api/peerconnection/RTCMediaConstraints.h"
#import "helpers/NSString+StdString.h"

@interface RTCMediaConstraintsTest : NSObject
- (void)testMediaConstraints;
@end

@implementation RTCMediaConstraintsTest

- (void)testMediaConstraints {
  NSDictionary *mandatory = @{@"key1": @"value1", @"key2": @"value2"};
  NSDictionary *optional = @{@"key3": @"value3", @"key4": @"value4"};

  RTCMediaConstraints *constraints = [[RTCMediaConstraints alloc]
      initWithMandatoryConstraints:mandatory
               optionalConstraints:optional];
  std::unique_ptr<webrtc::MediaConstraints> nativeConstraints =
      [constraints nativeConstraints];

  webrtc::MediaConstraints::Constraints nativeMandatory = nativeConstraints->GetMandatory();
  [self expectConstraints:mandatory inNativeConstraints:nativeMandatory];

  webrtc::MediaConstraints::Constraints nativeOptional = nativeConstraints->GetOptional();
  [self expectConstraints:optional inNativeConstraints:nativeOptional];
}

- (void)expectConstraints:(NSDictionary *)constraints
      inNativeConstraints:(webrtc::MediaConstraints::Constraints)nativeConstraints {
  EXPECT_EQ(constraints.count, nativeConstraints.size());

  for (NSString *key in constraints) {
    NSString *value = [constraints objectForKey:key];

    std::string nativeValue;
    bool found = nativeConstraints.FindFirst(key.stdString, &nativeValue);
    EXPECT_TRUE(found);
    EXPECT_EQ(value.stdString, nativeValue);
  }
}

@end

TEST(RTCMediaConstraintsTest, MediaConstraintsTest) {
  @autoreleasepool {
    RTCMediaConstraintsTest *test = [[RTCMediaConstraintsTest alloc] init];
    [test testMediaConstraints];
  }
}
