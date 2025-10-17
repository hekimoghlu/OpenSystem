/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
//  xc_cintltst.m
//  xc_cintltst
//
//  Created by Christopher Chapman on 2024-09-25.
//  Copyright Â© 2024 Apple. All rights reserved.
//

#import <XCTest/XCTest.h>
#import "ctest.h"

extern int main(int argc, const char* const argv[]);

int call_main(const char *argv1) {
    resetRepeat();
    char argv0[] = "intltest";
    int argc = 2;
    const char* const argv[] = {argv0, argv1, NULL};
    int result = main(argc, argv);
    return result;
}


@interface xc_cintltst : XCTestCase

@end

@implementation xc_cintltst

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

// TODO: implement dynamic tests: https://docs.apple.com/access/general/documentation/xctestinternal/060-dynamic-tests


- (void)test_complex {
    char argv1[] = "/complex";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_custrtrn {
    char argv1[] = "/custrtrn";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_hpmufn {
    char argv1[] = "/hpmufn";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_idna {
    char argv1[] = "/idna";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_putiltst {
    char argv1[] = "/putiltst";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_regex {
    char argv1[] = "/regex";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_spreptst {
    char argv1[] = "/spreptst";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_tscoll {
    char argv1[] = "/tscoll";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_tsconv {
    char argv1[] = "/tsconv";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_tsformat {
    char argv1[] = "/tsformat";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_tsformat_TestUAMeasureFormat {
    char argv1[] = "/tsformat/cmeasureformattest/TestUAMeasureFormat";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_tsformat_everything_else {
    const char* const argv[] = {
        "cintltst",
        "/tsformat/ccaltst",
        "/tsformat/cdateintervalformattest",
        "/tsformat/cdattst",
        "/tsformat/cdtdptst",
        "/tsformat/cdtrgtst",
        "/tsformat/cgendtst",

//        "/tsformat/cmeasureformattest",
//        "/tsformat/cmeasureformattest/TestUAMeasureFormat",
        "/tsformat/cmeasureformattest/TestUAMeasFmtOpenAllLocs",
        "/tsformat/cmeasureformattest/TestUAGetUnitsForUsage",
        "/tsformat/cmeasureformattest/TestUAGetCategoryForUnit",
        "/tsformat/cmeasureformattest/TestMeasurementSystemOverride",
        "/tsformat/cmeasureformattest/TestRgSubtag",

        "/tsformat/cmsgtst",
        "/tsformat/cnmdptst",
        "/tsformat/cnumtst",
        "/tsformat/cpluralrulestest",
        "/tsformat/crelativedateformattest",
        "/tsformat/currtest",
        "/tsformat/uatimezonetest",
        "/tsformat/udatpg_test",
        "/tsformat/uformattedvalue",
        "/tsformat/ulistfmttest",
        "/tsformat/unumberformatter",
        "/tsformat/unumberrangeformatter",
        "/tsformat/uregiontest",
        "/tsformat/utmstest",
    };
    resetRepeat();
    int argc = 26;
    int result = main(argc, argv);
    XCTAssertEqual(result, 0);
}

- (void)test_tsnorm {
    char argv1[] = "/tsnorm";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_tstformat {
    char argv1[] = "/tstformat";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_tstxtbd {
    char argv1[] = "/tstxtbd";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_tsutil {
    char argv1[] = "/tsutil";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_ucsdetst {
    char argv1[] = "/ucsdetst";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_udatatst {
    char argv1[] = "/udatatst";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_uset {
    char argv1[] = "/uset";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_uspoof {
    char argv1[] = "/uspoof";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_utf16tst {
    char argv1[] = "/utf16tst";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_utf8tst {
    char argv1[] = "/utf8tst";
    int result = call_main(argv1);
    XCTAssertEqual(result, 0);
}

- (void)test_utrans {
    char argv1[] = "/utrans";
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
