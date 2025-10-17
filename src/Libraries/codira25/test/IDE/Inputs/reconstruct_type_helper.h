/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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

#include "language_name.h"

typedef int Unwrapped;
typedef int Wrapped __attribute__((language_wrapper(struct)));

typedef int TagTypedefCollision __attribute__((language_name("TTCollisionTypedef")));
enum TagTypedefCollision {
  TagTypedefCollisionX
} __attribute__((language_name("TTCollisionTag")));


enum EnumByTag {
  EnumByTagX
};
typedef enum {
  EnumByTypedefX
} EnumByTypedef;
typedef enum EnumByBoth {
  EnumByBothX
} EnumByBoth;


#if __OBJC__
#include "language_name_objc.h"

@compatibility_alias SomeClassAlias SNSomeClass;

extern NSString *SomeErrorDomain;
enum SomeError {
  SomeErrorBadness
} __attribute__((ns_error_domain(SomeErrorDomain)));

extern NSString *SomeOtherErrorDomain;
typedef enum __attribute__((ns_error_domain(SomeOtherErrorDomain))) {
  SomeOtherErrorBadness
} SomeOtherError __attribute__((language_name("SomeRenamedError")));

#endif // __OBJC__

