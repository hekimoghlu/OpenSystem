/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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

#ifdef __OBJC__
#pragma clang assume_nonnull begin

@interface NewlyGenericSub<Element> : Base
+ (Element)defaultElement;
@end

@interface RenamedGeneric<Element: Base *> : Base
@end

@interface ClassWithManyRenames : Base
+ (instancetype)classWithManyRenamesForInt:(int)value;
- (instancetype)initWithBoolean:(_Bool)value __attribute__((language_name("init(finalBoolean:)")));

- (void)doImportantThings __attribute__((language_name("finalDoImportantThings()")));
@property (class, nullable) id importantClassProperty __attribute__((language_name("finalClassProperty")));
@property (nullable) id importantInstanceProperty __attribute__((language_name("finalInstanceProperty")));
@end

@interface PrintingRenamed : Base
- (void)print;
- (void)print:(id)thing;
- (void)print:(id)thing options:(id)options;

+ (void)print;
+ (void)print:(id)thing;
+ (void)print:(id)thing options:(id)options;
@end

@interface PrintingInterference : Base
- (void)print:(id)thing; // Only this one gets renamed.
- (void)print:(id)thing extra:(id)options;
@end

#pragma clang assume_nonnull end
#endif // __OBJC__
