/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#include "NestedClangTypesHelper.h"

struct Outer {
  int value;
};

struct __attribute__((language_name("Outer.InterestingValue"))) Inner {
  int value;
};

struct OuterFromOtherModule;
struct __attribute__((language_name("OuterFromOtherModule.InterestingValue"))) InnerCrossModule {
  int value;
};


#if __OBJC__

@import Foundation;

extern NSString * const ErrorCodeDomain;
enum __attribute__((ns_error_domain(ErrorCodeDomain))) ErrorCodeEnum {
  ErrorCodeEnumA
};

#endif // __OBJC__
