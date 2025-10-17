/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_ACCESS_SPECIFIERS_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_ACCESS_SPECIFIERS_H

class PublicPrivate {
public:
  int PublicMemberVar;
  static int PublicStaticMemberVar;
  void publicMemberFunc();

  typedef int PublicTypedef;
  struct PublicStruct {};
  enum PublicEnum { PublicEnumValue1 };
  enum { PublicAnonymousEnumValue1 };
  enum PublicClosedEnum {
    PublicClosedEnumValue1
  } __attribute__((enum_extensibility(closed)));
  enum PublicOpenEnum {
    PublicOpenEnumValue1
  } __attribute__((enum_extensibility(open)));
  enum PublicFlagEnum {} __attribute__((flag_enum));

private:
  int PrivateMemberVar;
  static int PrivateStaticMemberVar;
  void privateMemberFunc() {}

  typedef int PrivateTypedef;
  struct PrivateStruct {};
  enum PrivateEnum { PrivateEnumValue1 };
  enum { PrivateAnonymousEnumValue1 };
  enum PrivateClosedEnum {
    PrivateClosedEnumValue1
  } __attribute__((enum_extensibility(closed)));
  enum PrivateOpenEnum {
    PrivateOpenEnumValue1
  } __attribute__((enum_extensibility(open)));
  enum PrivateFlagEnum {} __attribute__((flag_enum));
};

#endif
