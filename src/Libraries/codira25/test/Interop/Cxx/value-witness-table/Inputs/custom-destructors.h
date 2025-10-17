/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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

#ifndef TEST_INTEROP_CXX_VALUE_WITNESS_TABLE_INPUTS_CUSTOM_DESTRUCTORS_H
#define TEST_INTEROP_CXX_VALUE_WITNESS_TABLE_INPUTS_CUSTOM_DESTRUCTORS_H

struct __attribute__((language_attr("import_unsafe"))) HasUserProvidedDestructor {
  int *value;
  HasUserProvidedDestructor() {}
  HasUserProvidedDestructor(int *value) : value(value) {}
#if __is_target_os(windows) && !defined(WIN_TRIVIAL)
  // On windows, force this type to be address-only.
  HasUserProvidedDestructor(const HasUserProvidedDestructor &other) : value(other.value) {}
#endif
  ~HasUserProvidedDestructor() { *value = 42; }
};

struct __attribute__((language_attr("import_unsafe")))
HasEmptyDestructorAndMemberWithUserDefinedConstructor {
  HasUserProvidedDestructor member;
  ~HasEmptyDestructorAndMemberWithUserDefinedConstructor() { /* empty */
  }
};

struct HasNonTrivialImplicitDestructor {
  HasUserProvidedDestructor member;
};

struct HasNonTrivialDefaultedDestructor {
  HasUserProvidedDestructor member;
  ~HasNonTrivialDefaultedDestructor() = default;
};

struct HasDefaultedDestructor {
  ~HasDefaultedDestructor() = default;
};

// For the following objects with virtual bases / destructors, make sure that
// any executable user of these objects disable rtti and exceptions. Otherwise,
// the linker will error because of undefined vtables.
// FIXME: Once we can link with libc++ we can enable RTTI.

struct __attribute__((language_attr("import_unsafe"))) HasVirtualBaseAndDestructor
    : virtual HasDefaultedDestructor {
  int *value;
  HasVirtualBaseAndDestructor(int *value) : value(value) {}
  ~HasVirtualBaseAndDestructor() { *value = 42; }
};

struct __attribute__((language_attr("import_unsafe"))) HasVirtualDestructor {
  // An object with a virtual destructor requires a delete operator in case
  // we try to delete the base object. Until we can link against libc++, use
  // this dummy implementation.
  static void operator delete(void *p) { __builtin_unreachable(); }
  virtual ~HasVirtualDestructor(){};
};

struct __attribute__((language_attr("import_unsafe")))
HasVirtualDefaultedDestructor {
  static void operator delete(void *p) { __builtin_unreachable(); }
  virtual ~HasVirtualDefaultedDestructor() = default;
};

struct __attribute__((language_attr("import_unsafe"))) HasBaseWithVirtualDestructor
    : HasVirtualDestructor {
  int *value;
  HasBaseWithVirtualDestructor(int *value) : value(value) {}
  ~HasBaseWithVirtualDestructor() { *value = 42; }
};

struct __attribute__((language_attr("import_unsafe")))
HasVirtualBaseWithVirtualDestructor : virtual HasVirtualDestructor {
  int *value;
  HasVirtualBaseWithVirtualDestructor(int *value) : value(value) {}
  ~HasVirtualBaseWithVirtualDestructor() { *value = 42; }
};

struct DummyStruct {};

struct __attribute__((language_attr("import_unsafe")))
HasUserProvidedDestructorAndDummy {
  DummyStruct dummy;
  HasUserProvidedDestructorAndDummy(DummyStruct dummy) : dummy(dummy) {}
#if __is_target_os(windows) && !defined(WIN_TRIVIAL)
  // On windows, force this type to be address-only.
  HasUserProvidedDestructorAndDummy(const HasUserProvidedDestructorAndDummy &other) : dummy(other.dummy) {}
#endif
  ~HasUserProvidedDestructorAndDummy() {}
};

// Make sure that we don't crash on struct templates with destructors.
template <typename T>
struct __attribute__((language_attr("import_unsafe"))) S {
  ~S() {}
};

#endif // TEST_INTEROP_CXX_VALUE_WITNESS_TABLE_INPUTS_CUSTOM_DESTRUCTORS_H
