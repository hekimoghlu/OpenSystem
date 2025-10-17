/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_DESTRUCTORS_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_DESTRUCTORS_H

struct
    __attribute__((language_attr("conforms_to:CodiraTest.Testable")))
    HasTest {
  void test() const;
};

struct
    __attribute__((language_attr("conforms_to:CodiraTest.Playable")))
    __attribute__((language_attr("import_reference")))
    __attribute__((language_attr("retain:immortal")))
    __attribute__((language_attr("release:immortal")))
    HasPlay {
  void play() const;
};

struct __attribute__((language_attr("conforms_to:CodiraTest.Testable")))
__attribute__((language_attr(
    "conforms_to:CodiraTest.Playable"))) MultipleConformanceHasTestAndPlay {
  void test() const;
  void play() const;
};

struct
    __attribute__((language_attr("conforms_to:ImportedModule.ProtocolFromImportedModule")))
    HasImportedConf {
  void testImported() const;
};

struct DerivedFromHasTest : HasTest {};
struct DerivedFromDerivedFromHasTest : HasTest {};
struct DerivedFromMultipleConformanceHasTestAndPlay
    : MultipleConformanceHasTestAndPlay {};

struct __attribute__((language_attr("conforms_to:CodiraTest.Testable")))
DerivedFromDerivedFromHasTestWithDuplicateArg : HasTest {};

struct DerivedFromHasPlay : HasPlay {};
struct DerivedFromDerivedFromHasPlay : HasPlay {};

struct HasTestAndPlay : HasPlay, HasTest {};
struct DerivedFromHasTestAndPlay : HasPlay, HasTest {};

struct DerivedFromHasImportedConf : HasImportedConf {};
struct DerivedFromDerivedFromHasImportedConf : HasImportedConf {};

#endif // TEST_INTEROP_CXX_CLASS_INPUTS_DESTRUCTORS_H
