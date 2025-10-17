/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#import "components/video_codec/RTCH264ProfileLevelId.h"

#import <XCTest/XCTest.h>

@interface RTCH264ProfileLevelIdTests : XCTestCase

@end

static NSString *level31ConstrainedHigh = @"640c1f";
static NSString *level31ConstrainedBaseline = @"42e01f";

@implementation RTCH264ProfileLevelIdTests

- (void)testInitWithString {
  RTCH264ProfileLevelId *profileLevelId =
      [[RTCH264ProfileLevelId alloc] initWithHexString:level31ConstrainedHigh];
  XCTAssertEqual(profileLevelId.profile, RTCH264ProfileConstrainedHigh);
  XCTAssertEqual(profileLevelId.level, RTCH264Level3_1);

  profileLevelId = [[RTCH264ProfileLevelId alloc] initWithHexString:level31ConstrainedBaseline];
  XCTAssertEqual(profileLevelId.profile, RTCH264ProfileConstrainedBaseline);
  XCTAssertEqual(profileLevelId.level, RTCH264Level3_1);
}

- (void)testInitWithProfileAndLevel {
  RTCH264ProfileLevelId *profileLevelId =
      [[RTCH264ProfileLevelId alloc] initWithProfile:RTCH264ProfileConstrainedHigh
                                               level:RTCH264Level3_1];
  XCTAssertEqualObjects(profileLevelId.hexString, level31ConstrainedHigh);

  profileLevelId = [[RTCH264ProfileLevelId alloc] initWithProfile:RTCH264ProfileConstrainedBaseline
                                                            level:RTCH264Level3_1];
  XCTAssertEqualObjects(profileLevelId.hexString, level31ConstrainedBaseline);
}

@end
