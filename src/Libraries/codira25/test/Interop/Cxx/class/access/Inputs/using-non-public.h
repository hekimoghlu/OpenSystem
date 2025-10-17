/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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

#ifndef _USING_BASE_METHODS_H
#define _USING_BASE_METHODS_H

enum class Return {
  publUsingPubl,
  protUsingPubl,
  omitUsingPubl,
  publUsingProt,
  protUsingProt,
  omitUsingProt,
};

class Base {
protected:
  Return omitUsingProt(void) const { return Return::omitUsingProt; }
  Return protUsingProt(void) const { return Return::protUsingProt; }
  Return publUsingProt(void) const { return Return::publUsingProt; }
public:
  Return omitUsingPubl(void) const { return Return::omitUsingPubl; }
  //     ^^^^     ^^^^
  //        |         ` access level of base method
  //         ` access level of using statement
  Return protUsingPubl(void) const { return Return::protUsingPubl; }
  Return publUsingPubl(void) const { return Return::publUsingPubl; }

  // N.B. private members should never be visible through inheritance, so having
  // `using Base::privateMethod` in a derived class should be a C++ error.
  //
  // Thus we'll forgo defining any private members here and test those
  // separately.
};

class PublUser : public Base {
public:
  using Base::publUsingProt;
  using Base::publUsingPubl;

protected:
  using Base::protUsingProt;
  using Base::protUsingPubl;
};

class ProtUser : protected Base {
public:
  using Base::publUsingProt;
  using Base::publUsingPubl;

protected:
  using Base::protUsingProt;
  using Base::protUsingPubl;
};

class PrivUser : private Base {
public:
  using Base::publUsingProt;
  using Base::publUsingPubl;

protected:
  using Base::protUsingProt;
  using Base::protUsingPubl;
};

class Publ : public Base {};
class Prot : protected Base {};
class Priv : private Base {};

// N.B. with another layer of inheritance, we could test *many* combinations,
// i.e.:
//
//   // Classes inheriting from a class with using decls
//   class _ : public/protected/private Publ/Prot/PrivUser {}
//
//   // Classes with using decls that skip past an inherited class
//   class _ : public/protected/private Publ/Prot/Priv {}
//
// but most combinations will lead to most/all fields being inaccessible.
//
// For now, we just spot-check two interesting cases, where publUsing*() should
// still be public in the derived class.
class PublPrivUser : public PrivUser {};

class PrivUserPubl : private Publ {
public:
  using Base::publUsingProt;
  using Base::publUsingPubl;

protected:
  using Base::protUsingProt;
  using Base::protUsingPubl;
};

#endif // _USING_BASE_METHODS_H
