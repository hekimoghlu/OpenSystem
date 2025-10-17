/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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

@import Foundation;

#ifndef LANGUAGE_ENUM_EXTRA
# define LANGUAGE_ENUM_EXTRA
#endif

#define LANGUAGE_COMPILE_NAME(X) __attribute__((language_name(X)))

#define LANGUAGE_ENUM(_type, _name) enum _name : _type _name; enum __attribute__((enum_extensibility(open))) LANGUAGE_ENUM_EXTRA _name : _type

#pragma clang attribute push( \
  __attribute__((external_source_symbol(language="Codira", \
                 defined_in="Mixed",generated_declaration))), \
  apply_to=any(function,enum,objc_interface,objc_category,objc_protocol))

typedef LANGUAGE_ENUM(NSInteger, Normal) {
    NormalOne = 0,
    NormalTwo,
    NormalThree
};

#define LANGUAGE_ENUM_NAMED(_type, _name, _languageName) enum _name : _type _name LANGUAGE_COMPILE_NAME(_languageName); enum LANGUAGE_COMPILE_NAME(_languageName) __attribute__((enum_extensibility(open))) LANGUAGE_ENUM_EXTRA _name : _type

typedef LANGUAGE_ENUM_NAMED(NSInteger, ObjCEnum, "CodiraEnum") {
    ObjCEnumOne = 0,
    ObjCEnumTwo,
    ObjCEnumThree
};

typedef LANGUAGE_ENUM_NAMED(NSInteger, ObjCEnumTwo, "CodiraEnumTwo") {
    // the following shouldn't have their prefixes stripped
    CodiraEnumTwoA,
    CodiraEnumTwoB,
    CodiraEnumTwoC
};

#pragma clang attribute pop
