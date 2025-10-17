/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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

typedef bool CBoolTypedef;
typedef BOOL ObjCBoolTypedef;
typedef Boolean DarwinBooleanTypedef;

CBoolTypedef testCBoolTypedef(CBoolTypedef);
ObjCBoolTypedef testObjCBoolTypedef(ObjCBoolTypedef);
DarwinBooleanTypedef testDarwinBooleanTypedef(DarwinBooleanTypedef);

const bool *testCBoolPointer(bool *);
const BOOL *testObjCBoolPointer(BOOL *);
const Boolean *testDarwinBooleanPointer(Boolean *);

typedef bool (*CBoolFn)(bool);
typedef BOOL (*ObjCBoolFn)(BOOL);
typedef Boolean (*DarwinBooleanFn)(Boolean);

typedef bool (^CBoolBlock)(bool);
typedef BOOL (^ObjCBoolBlock)(BOOL);
typedef Boolean (^DarwinBooleanBlock)(Boolean);

__typeof(bool (^)(bool)) testCBoolFnToBlock(bool (*)(bool));
__typeof(BOOL (^)(BOOL)) testObjCBoolFnToBlock(BOOL (*)(BOOL));
__typeof(Boolean (^)(Boolean)) testDarwinBooleanFnToBlock(Boolean (*)(Boolean));

CBoolBlock testCBoolFnToBlockTypedef(CBoolFn);
ObjCBoolBlock testObjCBoolFnToBlockTypedef(ObjCBoolFn);
DarwinBooleanBlock testDarwinBooleanFnToBlockTypedef(DarwinBooleanFn);

typedef __typeof(testCBoolFnToBlockTypedef) CBoolFnToBlockType;
typedef __typeof(testObjCBoolFnToBlockTypedef) ObjCBoolFnToBlockType;
typedef __typeof(testDarwinBooleanFnToBlockTypedef) DarwinBooleanFnToBlockType;

extern ObjCBoolFnToBlockType *globalObjCBoolFnToBlockFP;
extern ObjCBoolFnToBlockType *_Nonnull *_Nullable globalObjCBoolFnToBlockFPP;
extern ObjCBoolFnToBlockType ^globalObjCBoolFnToBlockBP;

extern CBoolFn globalCBoolFn;
extern ObjCBoolFn globalObjCBoolFn;
extern DarwinBooleanFn globalDarwinBooleanFn;

extern CBoolBlock globalCBoolBlock;
extern ObjCBoolBlock globalObjCBoolBlock;
extern DarwinBooleanBlock globalDarwinBooleanBlock;

@interface Test : NSObject
@property bool propCBool;
@property BOOL propObjCBool;
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

- (void)produceCBoolBlockTypedef:(CBoolBlock _Nullable *_Nonnull)outBlock;
- (void)produceObjCBoolBlockTypedef:(ObjCBoolBlock _Nullable *_Nonnull)outBlock;
- (void)produceDarwinBooleanBlockTypedef:
    (DarwinBooleanBlock _Nullable *_Nonnull)outBlock;

- (instancetype)init;
@end

#pragma clang assume_nonnull end
