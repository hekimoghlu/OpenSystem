/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_DEFINE_REFERENCED_INLINE_TYPES_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_DEFINE_REFERENCED_INLINE_TYPES_H

template <class T>
inline void inlineFn1(T) { }

template <class T>
inline void inlineFn2(T) { }

inline void inlineFn3() { }

template <class T>
struct __attribute__((language_attr("import_unsafe"))) HasInlineDtor {
  inline ~HasInlineDtor() { inlineFn1(T()); }
};

template<class T>
struct CtorCallsInlineFn {
  CtorCallsInlineFn(T x) { inlineFn2(x); }
};

template<class T>
struct HasInlineStaticMember {
  inline static void member() { inlineFn3(); }
};

template<class T>
struct ChildWithInlineCtorDtor1 {
  inline ChildWithInlineCtorDtor1() { HasInlineStaticMember<T>::member(); }
  inline ~ChildWithInlineCtorDtor1() { HasInlineStaticMember<T>::member(); }
};

template <class T>
struct __attribute__((language_attr("import_unsafe"))) ChildWithInlineCtorDtor2 {
  inline ChildWithInlineCtorDtor2() { HasInlineStaticMember<T>::member(); }
  inline ~ChildWithInlineCtorDtor2() { HasInlineStaticMember<T>::member(); }
};

template <class T>
struct __attribute__((language_attr("import_unsafe")))
ParentWithChildWithInlineCtorDtor : ChildWithInlineCtorDtor1<T> {};

template<class T>
struct HolderWithChildWithInlineCtorDtor {
  ChildWithInlineCtorDtor2<T> x;
};

template <class T>
struct __attribute__((language_attr("import_unsafe"))) DtorCallsInlineMethod {
  inline void unique_name() {}

  ~DtorCallsInlineMethod() {
    unique_name();
  }
};

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_DEFINE_REFERENCED_INLINE_TYPES_H
