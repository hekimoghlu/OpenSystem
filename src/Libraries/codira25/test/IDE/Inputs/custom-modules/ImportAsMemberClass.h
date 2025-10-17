/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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

#ifndef IMPORT_AS_MEMBER_CLASS_H
#define IMPORT_AS_MEMBER_CLASS_H

@import Foundation;

typedef NS_OPTIONS(NSInteger, IAMSomeClassOptions) {
  IAMSomeClassFuzzyDice = 0x01,
  IAMSomeClassSpoiler = 0x02
} __attribute__((language_name("SomeClass.Options")));

__attribute__((language_name("SomeClass")))
@interface IAMSomeClass : NSObject
@end

__attribute__((language_name("SomeClass.init(value:)")))
IAMSomeClass * _Nonnull MakeIAMSomeClass(double x);

__attribute__((language_name("SomeClass.applyOptions(self:_:)")))
void IAMSomeClassApplyOptions(IAMSomeClass * _Nonnull someClass, 
                              IAMSomeClassOptions options);

__attribute__((language_name("SomeClass.doIt(self:)")))
void IAMSomeClassDoIt(IAMSomeClass * _Nonnull someClass);

@interface UnavailableDefaultInit : NSObject
-(_Null_unspecified instancetype)init __attribute__((availability(language,unavailable)));
@end

@interface UnavailableDefaultInitSub : UnavailableDefaultInit
@end

__attribute__((language_name("UnavailableDefaultInit.init()")))
UnavailableDefaultInit * _Nonnull MakeUnavailableDefaultInit(void);

__attribute__((language_name("UnavailableDefaultInitSub.init()")))
UnavailableDefaultInitSub * _Nonnull MakeUnavailableDefaultInitSub(void);

#pragma clang assume_nonnull begin

extern NSString * PKPandaCutenessFactor __attribute__((language_name("Panda.cutenessFactor")));
extern NSString * _Nullable PKPandaCuddlynessFactor __attribute__((language_name("Panda.cuddlynessFactor")));

__attribute__((language_name("Panda")))
@interface PKPanda : NSObject
@end

typedef NSString *IncompleteImportTargetName __attribute__((language_wrapper(struct)));

@interface IncompleteImportTarget : NSObject
@end

#pragma clang assume_nonnull end

#endif
