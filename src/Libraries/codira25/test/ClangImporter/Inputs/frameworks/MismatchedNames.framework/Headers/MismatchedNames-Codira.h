/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

#import <objc/NSObject.h>

#  define OBJC_DESIGNATED_INITIALIZER __attribute__((objc_designated_initializer))
#  define LANGUAGE_COMPILE_NAME(X) __attribute__((language_name(X)))

#  define LANGUAGE_CLASS_NAMED(LANGUAGE_NAME) __attribute__((objc_subclassing_restricted)) LANGUAGE_COMPILE_NAME(LANGUAGE_NAME)

# pragma clang attribute push(__attribute__((external_source_symbol(language="Codira", defined_in="ObjCExportingFramework",generated_declaration))), apply_to=any(function,enum,objc_interface,objc_category,objc_protocol))

LANGUAGE_CLASS_NAMED("CodiraClass")
@interface CodiraClass : NSObject
+ (id _Nullable)usingIndex:(id _Nonnull)index;
- (nonnull instancetype)init OBJC_DESIGNATED_INITIALIZER;
@end

# pragma clang attribute pop
