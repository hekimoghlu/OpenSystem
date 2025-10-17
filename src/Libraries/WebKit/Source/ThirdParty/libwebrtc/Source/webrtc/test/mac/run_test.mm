/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#import <Cocoa/Cocoa.h>

#include "test/run_test.h"

// Converting a C++ function pointer to an Objective-C block.
typedef void(^TestBlock)();
TestBlock functionToBlock(void(*function)()) {
  return [^(void) { function(); } copy];
}

// Class calling the test function on the platform specific thread.
@interface TestRunner : NSObject {
  BOOL running_;
}
- (void)runAllTests:(TestBlock)ignored;
- (BOOL)running;
@end

@implementation TestRunner
- (id)init {
  self = [super init];
  if (self) {
    running_ = YES;
  }
  return self;
}

- (void)runAllTests:(TestBlock)testBlock {
  @autoreleasepool {
    testBlock();
    running_ = NO;
  }
}

- (BOOL)running {
  return running_;
}
@end

namespace webrtc {
namespace test {

void RunTest(void(*test)()) {
  @autoreleasepool {
    [NSApplication sharedApplication];

    // Convert the function pointer to an Objective-C block and call on a
    // separate thread, to avoid blocking the main thread.
    TestRunner *testRunner = [[TestRunner alloc] init];
    TestBlock testBlock = functionToBlock(test);
    [NSThread detachNewThreadSelector:@selector(runAllTests:)
                             toTarget:testRunner
                           withObject:testBlock];

    NSRunLoop *runLoop = [NSRunLoop currentRunLoop];
    while ([testRunner running] && [runLoop runMode:NSDefaultRunLoopMode
                                         beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.1]])
      ;
  }
}

}  // namespace test
}  // namespace webrtc
