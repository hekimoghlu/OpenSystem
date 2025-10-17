/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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

int barFunc1(int a);

int redeclaredInMultipleModulesFunc1(int a);

int barGlobalFunc(int a);

extern int barGlobalVariable;

extern int barGlobalVariableOldEnumElement;

int barGlobalFuncOldName(int a);

int barGlobalHoistedFuncOldName(int a, int b, int c);

@interface BarForwardDeclaredClass
- (id _Nonnull)initWithOldLabel0:(int)frame;
- (void) barInstanceFunc0;
- (void) barInstanceFunc1:(int)info anotherValue:(int)info1 anotherValue1:(int)info2 anotherValue2:(int)info3;
- (void) barInstanceFunc2:(int)info toRemove:(int)info1 toRemove1:(int)info2 toRemove2:(int)info3;
@end

enum BarForwardDeclaredEnum {
  BarForwardDeclaredEnumValue = 42
};

@interface PropertyUserInterface
- (int) field;
- (int * _Nullable) field2;
- (void) setField:(int)info;
- (void) setURL:(int)url;
+ (int) fieldPlus;
+ (void) methodPlus:(int)info;
+ (void) methodPlus;
@end

#define BAR_MACRO_1 0

typedef struct {
  int count;
  int theSimpleOldName;
  int theSimpleOldNameNotToRename;
} SomeItemSet;

typedef SomeItemSet SomeEnvironment;

@protocol WillOverrideWithTypeChange
- (SomeItemSet)doThing:(SomeItemSet)thing;
@end

#define CF_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NS_ENUM(_type, _name) CF_ENUM(_type, _name)

typedef NS_ENUM(long, FooComparisonResult) {
  FooOrderedAscending = -1L,
  FooOrderedSame,
  FooOrderedDescending,
  FooOrderedMemberSame,
  FooOrderedMovedToGlobal,
};

@interface BarBase
@end
@interface BarBaseNested
@end
