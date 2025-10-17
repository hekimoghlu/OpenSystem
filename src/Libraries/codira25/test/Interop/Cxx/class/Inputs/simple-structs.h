/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_SIMPLE_STRUCTS_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_SIMPLE_STRUCTS_H

struct HasPrivateFieldsOnly {
private:
  int priv1;
  int priv2;

public:
  HasPrivateFieldsOnly(int i1, int i2) : priv1(i1), priv2(i2) {}
};

struct HasPublicFieldsOnly {
  int publ1;
  int publ2;

  HasPublicFieldsOnly(int i1, int i2) : publ1(i1), publ2(i2) {}
};

struct HasPrivatePublicProtectedFields {
private:
  int priv1;

public:
  int publ1;

protected:
  int prot1;

protected:
  int prot2;

private:
  int priv2;

public:
  int publ2;

  HasPrivatePublicProtectedFields(int i1, int i2, int i3, int i4, int i5,
                                  int i6)
      : priv1(i1), publ1(i2), prot1(i3), prot2(i4), priv2(i5),
        publ2(i6) {}
};

struct Outer {
private:
  HasPrivatePublicProtectedFields privStruct;

public:
  HasPrivatePublicProtectedFields publStruct;

  Outer() : privStruct(1, 2, 3, 4, 5, 6), publStruct(7, 8, 9, 10, 11, 12) {}
};

#endif
