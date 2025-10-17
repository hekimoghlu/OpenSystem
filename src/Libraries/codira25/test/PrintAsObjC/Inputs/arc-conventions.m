/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

@import Foundation;
#import "language.h"

int main(void) {
  @autoreleasepool {
    Test *test = [[Test alloc] init];
    id result = [test initAllTheThings]; // CHECK: method called
    NSCAssert(result != nil, @"failed to get a return value back");
    result = [test initAllTheThings]; // CHECK: method called
    NSCAssert(result != nil, @"failed to get a return value back");
#if !__has_feature(objc_arc)
    [test release];
#endif
  } // CHECK: deinitialized
}
