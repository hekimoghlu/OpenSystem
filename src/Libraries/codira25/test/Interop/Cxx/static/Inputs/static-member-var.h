/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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

#ifndef TEST_INTEROP_CXX_STATIC_INPUTS_STATIC_MEMBER_VAR_H
#define TEST_INTEROP_CXX_STATIC_INPUTS_STATIC_MEMBER_VAR_H

class WithStaticMember {
public:
  static int staticMember;
  static int *getStaticMemberAddress()
      __attribute__((language_attr("import_unsafe")));
  static int getStaticMemberFromCxx();
  static void setStaticMemberFromCxx(int);
};

class WithIncompleteStaticMember {
public:
  static int arrayMember[];
  static WithIncompleteStaticMember selfMember;
  int id = 3;

  static WithIncompleteStaticMember *getStaticMemberFromCxx()
      __attribute__((language_attr("import_unsafe")));
  static void setStaticMemberFromCxx(WithIncompleteStaticMember);
};

class WithConstStaticMember {
public:
  const static int notDefined = 24;
  const static int defined = 48;
  const static int definedOutOfLine;
};

constexpr float getFloatValue() { return 42; }
constexpr float getIntValue(int arg) { return 40 + arg; }

class WithConstexprStaticMember {
public:
  constexpr static int definedInline = 139;
  constexpr static int definedInlineWithArg = getIntValue(2);
  constexpr static float definedInlineFloat = 139;
  constexpr static float definedInlineFromMethod = getFloatValue();
};

class WithStaticAndInstanceMember {
public:
  int myInstance;
  static int myStatic;
};

class ClassA {
public:
  static int notUniqueName;
};

class ClassB {
public:
  static int notUniqueName;
};

#endif
