/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

//
//  xc_iotest.m
//  xc_iotest
//
//  Created by Christopher Chapman on 2024-10-30.
//  Copyright Â© 2024 Apple. All rights reserved.
//

#import <XCTest/XCTest.h>

extern int main(int argc, char* argv[]);

int call_main(char *argv1) {
    char argv0[] = "iotest";
    int argc = 2;
    char* argv[] = {argv0, argv1, NULL};
    int result = main(argc, argv);
    return result;
}

@interface xc_iotest : XCTestCase

@end

@implementation xc_iotest

//- (void)setUp {
//    // Put setup code here. This method is called before the invocation of each test method in the class.
//}

//- (void)tearDown {
//    // Put teardown code here. This method is called after the invocation of each test method in the class.
//}

//- (void)testExample {
//    char argv1[] = "-h";
//    int result = call_main(argv1);
//    XCTAssertEqual(result, 0);
//}

- (void)test_datadriv {
    char argv1[] = "/datadriv";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_file {
    char argv1[] = "/file";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_stream {
    char argv1[] = "/stream";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_string {
    char argv1[] = "/string";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_translit {
    char argv1[] = "/translit";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_ScanfMultipleIntegers {
    char argv1[] = "/ScanfMultipleIntegers";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

//- (void)testPerformanceExample {
//    // This is an example of a performance test case.
//    [self measureBlock:^{
//        // Put the code you want to measure the time of here.
//    }];
//}

@end
