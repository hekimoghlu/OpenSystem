/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#ifndef _TRUSTTEST_MACRO_CONVERSIONS_H_
#define _TRUSTTEST_MACRO_CONVERSIONS_H_

#import <XCTest/XCTest.h>

#define isnt(THIS, THAT, ...) do { XCTAssertNotEqual(THIS, THAT, __VA_ARGS__); } while(0)

#define is(THIS, THAT, ...) do { XCTAssertEqual(THIS, THAT, __VA_ARGS__); } while(0)

#define is_status(THIS, THAT, ...) is(THIS, THAT)

#define ok(THIS, ...) do { XCTAssert(THIS, __VA_ARGS__); } while(0)

#define ok_status(THIS, ...) do { XCTAssertEqual(THIS, errSecSuccess, __VA_ARGS__); } while(0)

#define fail(...) ok(0, __VA_ARGS__)

#endif /* _TRUSTTEST_MACRO_CONVERSIONS_H_ */
