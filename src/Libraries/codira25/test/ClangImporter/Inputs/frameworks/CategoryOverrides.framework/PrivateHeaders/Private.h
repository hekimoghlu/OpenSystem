/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

#import <CategoryOverrides/CategoryOverrides.h>

@interface MyBaseClass ()
@property (nonatomic, strong, nullable) Base *derivedMember;
@end

@interface MyColor ()
+ (MyColor * _Null_unspecified) systemRedColor;
@end

@protocol MyPrivateProtocol
- (SomeStruct) myStructure;
@end

@interface MyBaseClass () <MyPrivateProtocol>
@end

@interface Refinery ()
@property (nonatomic, readwrite) RefinedSugar sugar;
@end

@interface ExtraRefinery ()
- (void)setSugar:(RefinedSugar)sugar;
@end

@interface MyBaseClass () <NonNullProtocol>
@end

@interface MyDerivedClass () <ReadwriteProtocol>
@end
