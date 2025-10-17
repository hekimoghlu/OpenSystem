/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_ACCESS_INVERSION_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_ACCESS_INVERSION_H

/// A record whose public members expose private members
struct Leaky {
public:
  Leaky() {
  } // Apparently necessary to ensure constructor is unambiguous in Codira

private:
  typedef bool PrivateAlias;

  struct PrivateRec {
  public:
    void privateRecMethod() const {}
    static const bool PRIVATE_REC_CONST = true;
  };

  enum PrivateEnum { privateEnumMember };

  enum class PrivateEnumClass { privateEnumClassMember };

  static const bool PRIVATE_CONST = true;

  // These are used as return values in functions that return private types
  static PrivateAlias privateAliasVal;
  static PrivateRec privateRecVal;
  static PrivateEnum privateEnumVal;
  static PrivateEnumClass privateEnumClassVal;

public:
  typedef PrivateAlias AliasToPrivateAlias;
  typedef PrivateRec AliasToPrivateRec;
  typedef PrivateEnum AliasToPrivateEnum;
  typedef PrivateEnumClass AliasToPrivateEnumClass;

  struct RecWithPrivateAlias {
    PrivateAlias mem;
  };
  struct RecWithPrivateRec {
    PrivateRec mem;
  };
  struct RecWithPrivateEnum {
    PrivateEnum mem;
  };
  struct RecWithPrivateEnumClass {
    PrivateEnumClass mem;
  };
  struct RecWithPrivateConst {
    const bool mem = PRIVATE_CONST;
  };

  static PrivateAlias staticReturningPrivateAlias() { return privateAliasVal; }
  static PrivateRec staticReturningPrivateRec() { return privateRecVal; }
  static PrivateEnum staticReturningPrivateEnum() { return privateEnumVal; }
  static PrivateEnumClass staticReturningPrivateEnumClass() {
    return privateEnumClassVal;
  }

  static void staticTakingPrivateAlias(PrivateAlias p) {}
  static void staticTakingPrivateRec(PrivateRec p) {}
  static void staticTakingPrivateEnum(PrivateEnum p) {}
  static void staticTakingPrivateEnumClass(PrivateEnumClass p) {}

  PrivateAlias methodReturningPrivateAlias() const { return privateAliasVal; }
  PrivateRec methodReturningPrivateRec() const { return privateRecVal; }
  PrivateEnum methodReturningPrivateEnum() const { return privateEnumVal; }
  PrivateEnumClass methodReturningPrivateEnumClass() const {
    return privateEnumClassVal;
  }

  void methodTakingPrivateAlias(PrivateAlias p) const {}
  void methodTakingPrivateRec(PrivateRec p) const {}
  void methodTakingPrivateEnum(PrivateEnum p) const {}
  void methodTakingPrivateEnumClass(PrivateEnumClass p) const {}

  void defaultArgOfPrivateRec(PrivateRec a = privateRecVal) const {}
  void defaultArgOfPrivateEnum(PrivateEnum a = privateEnumMember) const {}
  void defaultArgOfPrivateEnumClass(
      PrivateEnumClass a = PrivateEnumClass::privateEnumClassMember) const {}
  void defaultArgOfPrivateConst(bool a = PRIVATE_CONST) const {}
  void
  defaultArgOfPrivateRecConst(bool a = PrivateRec::PRIVATE_REC_CONST) const {}
};

#endif
