/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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

// This file is meant to be included with modules turned off, compiled against
// the fake clang-importer-sdk.
#import <Foundation.h>

@interface Outer : NSObject
@end

__attribute__((language_name("Outer.Inner")))
@interface InnerClass : NSObject
@end

struct __attribute__((language_name("Outer.InnerV"))) InnerStruct {
  int value;
};

typedef struct {
  int value;
} InnerAnonStruct __attribute__((language_name("Outer.InnerAS")));

typedef int InnerAlias __attribute__((language_name("Outer.InnerA")));


@interface NullabilityBase : NSObject
- (nonnull instancetype)initFormerlyFailableValue:(NSInteger)value __attribute__((objc_designated_initializer));
- (void)processNowNullableArgument:(nullable NSObject *)object;
@end
