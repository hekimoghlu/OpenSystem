/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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

#ifndef NULLABILITY_H
#define NULLABILITY_H

@import Foundation;

_Nullable id getId1(void);

extern _Nullable id global_id;

@interface SomeClass
- (nonnull id)methodA:(nullable SomeClass *)obj;
- (nonnull id)methodB:(nullable int (^)(int, int))block;
- (nullable id)methodC;
@property (nullable) id property;
- (id)methodD __attribute__((returns_nonnull));
- (void)methodE:(SomeClass *) __attribute__((nonnull)) obj;
- (void)methodF:(SomeClass *)obj second:(SomeClass *)obj2 __attribute__((nonnull));
- (void)methodG:(SomeClass *)obj second:(SomeClass *)obj2 __attribute__((nonnull(1)));
-(nonnull NSString *)stringMethod;
-(nullable NSArray *)optArrayMethod;

+(nonnull instancetype)someClassWithInt:(int)x;
+(nullable SomeClass*)someClassWithDouble:(double)x;
-(nonnull instancetype)returnMe;

@property (null_resettable) NSString *defaultedProperty;

@property (nonnull) NSString *funnyProperty;
-(nullable NSString *)funnyProperty;
-(void)setFunnyProperty:(null_unspecified NSString *)value;
@end

#define NON_NULL_MACRO(...) __attribute__ ((nonnull(__VA_ARGS__)))

void compare_classes(SomeClass *sc1, SomeClass *sc2, SomeClass *sc3) NON_NULL_MACRO(1,2);

#endif
