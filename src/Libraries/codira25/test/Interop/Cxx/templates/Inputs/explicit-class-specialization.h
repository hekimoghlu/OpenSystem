/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_EXPLICIT_CLASS_SPECIALIZATION_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_EXPLICIT_CLASS_SPECIALIZATION_H

struct SpecializedIntWrapper {
  int value;
  int getValue() const { return value; }
};

struct NonSpecializedIntWrapper {
  int value;
  int getValue() const { return value; }
};

template <class T>
struct MagicWrapper {
  T t;
  int doubleIfSpecializedElseTriple() const { return 3 * t.getValue(); }
};

template <>
struct MagicWrapper<SpecializedIntWrapper> {
  SpecializedIntWrapper t;
  int doubleIfSpecializedElseTriple() const { return 2 * t.getValue(); }
};

typedef MagicWrapper<SpecializedIntWrapper> WrapperWithSpecialization;
typedef MagicWrapper<NonSpecializedIntWrapper> WrapperWithoutSpecialization;

// Make sure these declarations don't cause a crash even though we can't import
// them.

template <class...> class HasSpecializations;

template <> class HasSpecializations<> {
  int value;
  struct Child {};
  enum Maybe : int { No, Yes };
};

template <> class HasSpecializations<int> {
  int value;
  struct Child {};
  enum Maybe : int { No, Yes };
};

template <> class HasSpecializations<int, int> {
  int value;
  struct Child {};
  enum Maybe : int { No, Yes };
};

template <class T> class HasSpecializations<T, int> {
  int value;
  struct Child {};
  enum Maybe : int { No, Yes };
};

template <class T, class... Ts> class HasSpecializations<int, T, Ts...> {
  int value;
  struct Child {};
  enum Maybe : int { No, Yes };
};

template <class>
struct HasEmptySpecializationAndStaticDateMember {
  inline static const bool value = false;
};

template <>
struct HasEmptySpecializationAndStaticDateMember<char> {
  inline static const bool value = true;
};

using HasEmptySpecializationAndStaticDateMemberInt = HasEmptySpecializationAndStaticDateMember<int>;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_EXPLICIT_CLASS_SPECIALIZATION_H
