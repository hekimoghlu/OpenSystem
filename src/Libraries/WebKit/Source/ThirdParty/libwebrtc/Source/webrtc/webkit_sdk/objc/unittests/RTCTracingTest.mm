/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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

#include <vector>

#include "rtc_base/gunit.h"

#import "api/peerconnection/RTCTracing.h"
#import "helpers/NSString+StdString.h"

@interface RTCTracingTest : NSObject
- (void)tracingTestNoInitialization;
@end

@implementation RTCTracingTest

- (NSString *)documentsFilePathForFileName:(NSString *)fileName {
  NSParameterAssert(fileName.length);
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
  NSString *documentsDirPath = paths.firstObject;
  NSString *filePath =
  [documentsDirPath stringByAppendingPathComponent:fileName];
  return filePath;
}

- (void)tracingTestNoInitialization {
  NSString *filePath = [self documentsFilePathForFileName:@"webrtc-trace.txt"];
  EXPECT_EQ(NO, RTCStartInternalCapture(filePath));
  RTCStopInternalCapture();
}

@end

TEST(RTCTracingTest, TracingTestNoInitialization) {
  @autoreleasepool {
    RTCTracingTest *test = [[RTCTracingTest alloc] init];
    [test tracingTestNoInitialization];
  }
}
