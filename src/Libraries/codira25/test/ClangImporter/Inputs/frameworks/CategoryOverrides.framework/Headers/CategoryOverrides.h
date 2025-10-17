/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

__attribute__((objc_root_class))
@interface Base
- (nonnull instancetype)init;
@end

typedef struct SomeStruct_s {
  int inner;
} SomeStruct;

@interface MyColor : Base
@property (class, nonatomic, readonly) MyColor *systemRedColor;
@end

@interface MyBaseClass : Base
// @property (nonatomic, strong, nullable) Base *derivedMember;
@property (nonatomic, assign, readonly) SomeStruct myStructure;
@end

@interface MyDerivedClass : MyBaseClass
@property (nonatomic, strong, nullable) Base *derivedMember;
@end

typedef enum {
  Caster,
  Grantulated,
  Confectioners,
  Cane,
  Demerara,
  Turbinado,
} RefinedSugar /*NS_REFINED_FOR_LANGUAGE*/ __attribute__((language_private));

@interface Refinery : Base
@property (nonatomic, readonly) RefinedSugar sugar /*NS_REFINED_FOR_LANGUAGE*/ __attribute__((language_private));
@end

@interface ExtraRefinery : Base
@property (nonatomic, readonly) RefinedSugar sugar /*NS_REFINED_FOR_LANGUAGE*/ __attribute__((language_private));
@end

@protocol NullableProtocol
@property (nonatomic, readonly, nullable) Base *requirement;
@end

@protocol NonNullProtocol <NullableProtocol>
@property (nonatomic, readonly, nonnull) Base *requirement;
@end

@protocol ReadonlyProtocol
@property (nonatomic, readonly) int answer;
@end

@protocol ReadwriteProtocol <ReadonlyProtocol>
@property (nonatomic, readwrite) int answer;
@end
