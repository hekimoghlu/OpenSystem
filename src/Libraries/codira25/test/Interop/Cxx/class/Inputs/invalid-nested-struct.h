/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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

// When we import this class, make sure that we bail before trying to import
// its sub-decls (i.e., "ForwardDeclaredSibling").
struct CannotImport {
  void test(struct ForwardDeclaredSibling *) {}

  ~CannotImport() = delete;
  CannotImport(CannotImport const &) = delete;
};

// We shouldn't be able to import this class either because it also doesn't have
// a copy ctor or destructor.
struct ForwardDeclaredSibling : CannotImport {};

// This is a longer regression test to make sure we don't improperly cache a
// typedef that's invalid.
namespace RegressionTest {

template <class From>
struct pointer_traits {
  template <class To>
  struct rebind {
    typedef To other;
  };
};

template <class T, class U>
struct Convert {
  typedef typename pointer_traits<T>::template rebind<U>::other type;
};

template <class>
struct Forward;

template <class V>
struct SomeTypeTrait {
  typedef Forward<V> *F;
  typedef typename Convert<V, F>::type A;
};

template <class V>
struct Forward {
  typedef typename SomeTypeTrait<V>::A A;

private:
  ~Forward() = delete;
  Forward(Forward const &) = delete;
};

template <class V>
struct SomeOtherTypeTrait : SomeTypeTrait<V> {
  typedef typename SomeTypeTrait<V>::A A;
};

// Just to instantiate all the templates.
struct FinalUser {
  typedef Forward<void *> F;
  typedef SomeOtherTypeTrait<void *> O;
  typedef SomeTypeTrait<void *> Z;
};

void test(typename FinalUser::Z) {}

} // namespace RegressionTest
