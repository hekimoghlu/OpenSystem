/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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

@interface GenericClass<T>: Base
@end

@interface PropertiesInit : Base
@property (readonly, nonnull) Base *readwriteChange;
@property (readonly, nonnull) Base *nullabilityChange;
@property (readonly, nonnull) GenericClass<Base *> *missingGenerics;
@property (readonly, nonnull) Base *typeChange;
@end

@interface PropertiesNoInit : Base
@property (readonly, nonnull) Base *readwriteChange;
@property (readonly, nonnull) Base *nullabilityChange;
@property (readonly, nonnull) GenericClass<Base *> *missingGenerics;
@property (readonly, nonnull) Base *typeChange;
@end

@interface PropertiesInitGeneric<T> : Base
@property (readonly, nonnull) Base *readwriteChange;
@property (readonly, nonnull) Base *nullabilityChange;
@property (readonly, nonnull) GenericClass<Base *> *missingGenerics;
@property (readonly, nonnull) Base *typeChange;
@end

@interface PropertiesNoInitGeneric<T> : Base
@property (readonly, nonnull) Base *readwriteChange;
@property (readonly, nonnull) Base *nullabilityChange;
@property (readonly, nonnull) GenericClass<Base *> *missingGenerics;
@property (readonly, nonnull) Base *typeChange;
@end

@interface PropertiesInitCategory : Base
@end

@interface PropertiesInitCategory (Category)
@property (readonly, nonnull) Base *readwriteChange;
@property (readonly, nonnull) Base *nullabilityChange;
@property (readonly, nonnull) GenericClass<Base *> *missingGenerics;
@property (readonly, nonnull) Base *typeChange;
@end

@interface PropertiesNoInitCategory : Base
@end

@interface PropertiesNoInitCategory (Category)
@property (readonly, nonnull) Base *readwriteChange;
@property (readonly, nonnull) Base *nullabilityChange;
@property (readonly, nonnull) GenericClass<Base *> *missingGenerics;
@property (readonly, nonnull) Base *typeChange;
@end
