/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

//===-- Optional.h - Simple variant for passing optional values ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides Optional, a template class modeled in the spirit of
//  OCaml's 'opt' variant.  The idea is to strongly type whether or not
//  a value can be optional.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_OPTIONAL
#define LLVM_ADT_OPTIONAL

#include <cassert>

namespace llvm {

template<typename T>
class Optional {
  T x;
  unsigned hasVal : 1;
public:
  explicit Optional() : x(), hasVal(false) {}
  Optional(const T &y) : x(y), hasVal(true) {}

  static inline Optional create(const T* y) {
    return y ? Optional(*y) : Optional();
  }

  Optional &operator=(const T &y) {
    x = y;
    hasVal = true;
    return *this;
  }
  
  const T* getPointer() const { assert(hasVal); return &x; }
  const T& getValue() const { assert(hasVal); return x; }

  operator bool() const { return hasVal; }
  bool hasValue() const { return hasVal; }
  const T* operator->() const { return getPointer(); }
  const T& operator*() const { assert(hasVal); return x; }
};

template<typename T> struct simplify_type;

template <typename T>
struct simplify_type<const Optional<T> > {
  typedef const T* SimpleType;
  static SimpleType getSimplifiedValue(const Optional<T> &Val) {
    return Val.getPointer();
  }
};

template <typename T>
struct simplify_type<Optional<T> >
  : public simplify_type<const Optional<T> > {};

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator==(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator!=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator<(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator<=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator>=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose 
/// errors at compile time.
template<typename T, typename U>
void operator>(const Optional<T> &X, const Optional<U> &Y);

} // end llvm namespace

#endif
