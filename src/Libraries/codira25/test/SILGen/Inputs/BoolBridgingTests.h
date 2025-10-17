/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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

#pragma clang assume_nonnull begin

// stdbool.h uses #define, so this test does as well.
#ifndef bool
# define bool _Bool
#endif

bool testCBool(bool);
BOOL testObjCBool(BOOL);
Boolean testDarwinBoolean(Boolean);

typedef bool (*CBoolFn)(bool);
typedef BOOL (*ObjCBoolFn)(BOOL);
typedef Boolean (*DarwinBooleanFn)(Boolean);

typedef bool (^CBoolBlock)(bool);
typedef BOOL (^ObjCBoolBlock)(BOOL);
typedef Boolean (^DarwinBooleanBlock)(Boolean);

__typeof(bool (^)(bool)) testCBoolFnToBlock(bool (*)(bool));
__typeof(BOOL (^)(BOOL)) testObjCBoolFnToBlock(BOOL (*)(BOOL));
__typeof(Boolean (^)(Boolean)) testDarwinBooleanFnToBlock(Boolean (*)(Boolean));

@interface Test : NSObject
@property bool propCBool __attribute__((language_name("propCBool")));
@property BOOL propObjCBool __attribute__((language_name("propObjCBool")));
@property Boolean propDarwinBoolean;

- (bool)testCBool:(bool)b;
- (BOOL)testObjCBool:(BOOL)b;
- (Boolean)testDarwinBoolean:(Boolean)b;

@property bool (^propCBoolBlock)(bool);
@property BOOL (^propObjCBoolBlock)(BOOL);
@property Boolean (^propDarwinBooleanBlock)(Boolean);

- (bool (^)(bool))testCBoolFnToBlock:(bool (*)(bool))fp;
- (BOOL (^)(BOOL))testObjCBoolFnToBlock:(BOOL (*)(BOOL))fp;
- (Boolean (^)(Boolean))testDarwinBooleanFnToBlock:(Boolean (*)(Boolean))fp;

- (instancetype)init;
@end

#pragma clang assume_nonnull end
