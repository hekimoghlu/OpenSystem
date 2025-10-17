/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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

#ifndef _USING_BASE_MEMBERS_H
#define _USING_BASE_MEMBERS_H

struct PublicBase {
private:
  int value = 123;

public:
  int publicGetter() const { return value; }
  void publicSetter(int v) { value = v; }
  void notExposed() const {}
};

struct PublicBasePrivateInheritance : private PublicBase {
  using PublicBase::publicGetter;
  using PublicBase::publicSetter;
};

struct PublicBaseProtectedInheritance : protected PublicBase {
  using PublicBase::publicGetter;
  using PublicBase::publicSetter;
};

struct PublicBaseUsingPrivateTypedef : private PublicBase {
private:
  typedef PublicBase MyBase;
public:
  using MyBase::publicGetter;
  using MyBase::publicSetter;
};

struct PublicBaseUsingPrivateUsingType : private PublicBase {
private:
  using MyBase = PublicBase;
public:
  using MyBase::publicGetter;
  using MyBase::publicSetter;
};

struct IntBox {
  int value;
  IntBox(int value) : value(value) {}
  IntBox(unsigned value) : value(value) {}
};

struct UsingBaseConstructorWithParam : IntBox {
  using IntBox::IntBox;
};

struct Empty {};

struct UsingBaseConstructorEmpty : private Empty {
  using Empty::Empty;

  int value = 456;
};

struct ProtectedBase {
protected:
  int protectedGetter() const { return 456; }
};

struct ProtectedMemberPrivateInheritance : private ProtectedBase {
  using ProtectedBase::protectedGetter;
};

struct OperatorBase {
  operator bool() const { return true; }
  int operator*() const { return 456; }
  OperatorBase operator!() const { return *this; }
  // int operator[](const int x) const { return x; } // FIXME: see below
};

struct OperatorBasePrivateInheritance : private OperatorBase {
public:
  using OperatorBase::operator bool;
  using OperatorBase::operator*;
  using OperatorBase::operator!;
  // using OperatorBase::operator[];  // FIXME: using operator[] is broken
};

#endif // !_USING_BASE_MEMBERS_H
