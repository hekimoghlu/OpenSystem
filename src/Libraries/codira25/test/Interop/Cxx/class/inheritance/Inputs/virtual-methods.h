/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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

extern "C" void puts(const char *_Null_unspecified);

inline void testFunctionCollected() {
  puts("test\n");
}

struct Base {
  virtual void foo() = 0;
  virtual void virtualRename() const
      __attribute__((language_name("languageVirtualRename()")));
};

struct Base2 { virtual int f() = 0; };
struct Base3 { virtual int f() { return 24; } };
struct Derived2 : public Base2 { virtual int f() {  return 42; } };
struct Derived3 : public Base3 { virtual int f() {  return 42; } };
struct Derived4 : public Base3 { };
struct DerivedFromDerived2 : public Derived2 {};

template <class T>
struct Derived : Base {
  inline void foo() override {
    testFunctionCollected();
  }

  void callMe() {
  }
};

using DerivedInt = Derived<int>;

template <class T>
struct Unused : Base {
  inline void foo() override {
  }
};

using UnusedInt = Unused<int>;

struct VirtualNonAbstractBase {
  virtual void nonAbstractMethod() const;
};

struct CallsPureMethod {
  virtual int getPureInt() const = 0;
  int getInt() const { return getPureInt() + 1; }
};

struct DerivedFromCallsPureMethod : CallsPureMethod {
  int getPureInt() const override { return 789; }
};

struct DerivedFromDerivedFromCallsPureMethod : DerivedFromCallsPureMethod {};
