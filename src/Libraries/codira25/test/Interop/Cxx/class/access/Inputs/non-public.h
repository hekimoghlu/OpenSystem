/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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

#ifndef NON_PUBLIC_H
#define NON_PUBLIC_H

// Override this to test structs
#ifndef TEST_CLASS
#define TEST_CLASS class
#endif

// Override this to test protected
#ifndef TEST_PRIVATE
#define TEST_PRIVATE private
#endif

/// A C++ class with various kinds of public and non-public fields, all of which
/// should be imported. Non-public fields should only be accessible inside
/// MyClass extensions in blessed.code.
TEST_CLASS
__attribute__((__language_attr__("private_fileid:main/blessed.code"))) MyClass {

public:
  void publMethod(void) const {}
  void publMutatingMethod(void) {}
  int publVar;
  static void publStaticFunc(void) {};
  static inline int publStaticVar = 0;

  typedef int publTypedef;
  publTypedef publTypedefMake(void) const { return 42; }
  void publTypedefTake(publTypedef x) const { }

  struct publStruct { int x; };
  publStruct publStructMake(void) const { return publStruct{}; }
  void publStructTake(publStruct x) const { }

  enum publEnum { variantPublEnum };
  publEnum publEnumMake(void) const { return variantPublEnum; }
  void publEnumTake(publEnum x) const { }

  enum class publEnumClass { variantPublEnumClass };
  publEnumClass publEnumClassMake(void) const { return publEnumClass::variantPublEnumClass; }
  void publEnumClassTake(publEnumClass x) const { }

  enum { publEnumAnonValue1 };
  enum publEnumClosed {
    variantPublEnumClosed
  } __attribute__((enum_extensibility(closed)));
  enum publEnumOpen {
    variantPublEnumOpen
  } __attribute__((enum_extensibility(open)));
  enum publEnumFlag {} __attribute__((flag_enum));

TEST_PRIVATE:
  void privMethod(void) const {}
  void privMutatingMethod(void) {}
  int privVar;
  static void privStaticFunc(void) {};
  static inline int privStaticVar = 0;

  typedef int privTypedef;
  privTypedef privTypedefMake(void) const { return 42; }
  void privTypedefTake(privTypedef x) const { }

  struct privStruct { int x; };
  privStruct privStructMake(void) const { return privStruct{}; }
  void privStructTake(privStruct x) const { }

  enum privEnum { variantPrivEnum };
  privEnum privEnumMake(void) const { return variantPrivEnum; }
  void privEnumTake(privEnum x) const { }

  enum class privEnumClass { variantPrivEnumClass };
  privEnumClass privEnumClassMake(void) const { return privEnumClass::variantPrivEnumClass; }
  void privEnumClassTake(privEnumClass x) const { }

  enum { privEnumAnonValue1 };
  enum privEnumClosed {
    variantPrivEnumClosed
  } __attribute__((enum_extensibility(closed)));
  enum privEnumOpen {
    variantPrivEnumOpen
  } __attribute__((enum_extensibility(open)));
  enum privEnumFlag {} __attribute__((flag_enum));
};

/// A C++ templated class, whose non-public fields should be accessible in
/// extensions of the (instantiated) class in blessed.code.
template <typename T>
TEST_CLASS __attribute__((
    __language_attr__("private_fileid:main/blessed.code"))) MyClassTemplate {
public:
  T publMethodT(T t) const { return t; }
  T publVarT;
  typedef T publTypedefT;

  void publMethod(void) const {}
  int publVar;
  typedef int publTypedef;
TEST_PRIVATE:
  T privMethodT(T t) const { return t; }
  T privVarT;
  typedef T privTypedefT;

  void privMethod(void) const {}
  int privVar;
  typedef int privTypedef;
};

typedef MyClassTemplate<float> MyFloatyClass;
typedef MyClassTemplate<MyClass> MyClassyClass;

#endif /* NON_PUBLIC_H */
