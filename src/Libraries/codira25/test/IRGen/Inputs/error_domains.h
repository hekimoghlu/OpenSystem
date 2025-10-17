/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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

#define MY_ERROR_ENUM(_type, _name, _domain)                                   \
  enum _name : _type _name;                                                    \
  enum __attribute__((ns_error_domain(_domain))) _name : _type

@class NSString;

extern NSString * const TagDomain1;
typedef MY_ERROR_ENUM(int, TagError1, TagDomain1) {
  Badness
};

extern NSString * const TagDomain2;
typedef MY_ERROR_ENUM(int, TagError2, TagDomain2) {
  Sickness
};

extern NSString * const TypedefDomain1;
typedef enum __attribute__((ns_error_domain(TypedefDomain1))) {
  Wrongness
} TypedefError1;

extern NSString *TypedefDomain2;
typedef enum __attribute__((ns_error_domain(TypedefDomain2))) {
  Illness
} TypedefError2;

@interface Nested @end

extern NSString * const NestedTagDomain;
typedef MY_ERROR_ENUM(int, NestedTagError, NestedTagDomain) {
  Trappedness
} __attribute__((language_name("Nested.TagError")));

extern NSString *NestedTypedefDomain;
typedef enum __attribute__((ns_error_domain(NestedTypedefDomain))) {
  Brokenness
} NestedTypedefError __attribute__((language_name("Nested.TypedefError")));
