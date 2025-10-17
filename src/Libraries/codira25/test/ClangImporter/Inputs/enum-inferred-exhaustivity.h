/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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

#if defined(CF_ENUM)
# error "This test requires controlling the definition of CF_ENUM"
#endif

// Make this C-compatible by leaving out the type.
#define CF_ENUM(_name) enum _name _name; enum _name

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmicrosoft-enum-forward-reference"
typedef CF_ENUM(EnumWithDefaultExhaustivity) {
  EnumWithDefaultExhaustivityLoneCase
};

// This name is also specially recognized by Codira.
#define __CF_ENUM_ATTRIBUTES __attribute__((enum_extensibility(open)))
typedef CF_ENUM(EnumWithSpecialAttributes) {
  EnumWithSpecialAttributesLoneCase
} __CF_ENUM_ATTRIBUTES;
#pragma clang diagnostic pop
