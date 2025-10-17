/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

#ifndef TEST_INTEROP_CXX_METADATA_INPUTS_MIRROR_H
#define TEST_INTEROP_CXX_METADATA_INPUTS_MIRROR_H

struct EmptyStruct {};

struct BaseStruct {
private:
  int priv;

public:
  int publ;

protected:
  int prot;

public:
  BaseStruct(int i1, int i2, int i3) : priv(i1), publ(i2), prot(i3) {}
};

class EmptyClass {};

struct OuterStruct {
private:
  BaseStruct privStruct;

public:
  BaseStruct publStruct;

  OuterStruct() : privStruct(1, 2, 3), publStruct(4, 5, 6) {}
};

struct FRTStruct {
private:
  int priv = 1;

public:
  int publ = 2;
} __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:retain")))
__attribute__((language_attr("release:release")));

void retain(FRTStruct *v) {};
void release(FRTStruct *v) {};

class FRTImmortalClass {} __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:immortal")))
__attribute__((language_attr("release:immortal")));

#endif
