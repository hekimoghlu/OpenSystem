/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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

#ifndef TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_NULLABLE_H
#define TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_NULLABLE_H

struct IntHolder { int value; };

struct
    __attribute__((language_attr("import_reference")))
    __attribute__((language_attr("retain:immortal")))
    __attribute__((language_attr("release:immortal")))
    IntBase : IntHolder {
  int i;
};

struct NoDtorThreeByte {
  char x;
  char y;
  char z;
  ~NoDtorThreeByte() = delete;
};

struct
    __attribute__((language_attr("import_reference")))
    __attribute__((language_attr("retain:immortal")))
    __attribute__((language_attr("release:immortal")))
    IntCharRef {
  int i;
  char b;
};

struct
    __attribute__((language_attr("import_reference")))
    __attribute__((language_attr("retain:immortal")))
    __attribute__((language_attr("release:immortal")))
    IntCharValue {
  int i;
  char b;
};


struct
    __attribute__((language_attr("import_reference")))
    __attribute__((language_attr("retain:immortal")))
    __attribute__((language_attr("release:immortal")))
    UnimportableMemberRef {
  int z; int zz; NoDtorThreeByte x; NoDtorThreeByte xx; int y;
};

struct
    __attribute__((language_attr("import_reference")))
    __attribute__((language_attr("retain:immortal")))
    __attribute__((language_attr("release:immortal")))
    UnimportableMemberValue {
  int z; int zz; NoDtorThreeByte x; NoDtorThreeByte xx; int y;
};

#endif // TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_NULLABLE_H
