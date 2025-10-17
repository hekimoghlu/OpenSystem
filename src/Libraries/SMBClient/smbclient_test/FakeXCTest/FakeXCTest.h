/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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
//  FakeXCTest
//
//  Copyright (c) 2014 Apple. All rights reserved.
//

#import <Foundation/Foundation.h>

#include "FakeXCTestCase.h"

extern int (*XFakeXCTestCallback)(const char *fmt, va_list);

@interface XCTest : NSObject

+ (void)setLimit:(const char *)testLimit;
+ (int) runTests;

@end

void FakeXCFailureHandler(XCTestCase *, BOOL, const char *, NSUInteger, NSString *, NSString *, ...);

#define FakeXCFailure(test, format...)                  \
    (void) FakeXCFailureHandler(test, false, __FILE__, __LINE__, @"failure", @format)

#define XCTFail(format...)            \
    FakeXCFailure(self, format);

#define XCTAssert(expression, format...)                \
    @try {                                              \
        BOOL expressionValue = !!(expression);          \
        if (!expressionValue) {                         \
            FakeXCFailure(self, format);                \
        }                                               \
    }                                                   \
    @catch (...) {                                      \
        FakeXCFailure(self, format);                    \
    }

#define XCTAssertTrue(expression, format...)            \
    XCTAssert(expression, format)


#define XCTAssertNotEqual(e1, e2, format...)            \
    @try {                                              \
        __typeof(e1) ee1 = e1;                          \
        __typeof(e2) ee2 = e2;                          \
        if (ee1 == ee2) {                               \
            FakeXCFailure(self, format);                \
        }                                               \
    }                                                   \
    @catch (...) {                                      \
        FakeXCFailure(self, format);                    \
    }

#define XCTAssertEqual(e1, e2, format...)               \
    @try {                                              \
        __typeof(e1) ee1 = e1;                          \
        __typeof(e2) ee2 = e2;                          \
        if (ee1 != ee2) {                               \
            FakeXCFailure(self, format);                \
        }                                               \
    }                                                   \
    @catch (...) {                                      \
        FakeXCFailure(self, format);                    \
    }

