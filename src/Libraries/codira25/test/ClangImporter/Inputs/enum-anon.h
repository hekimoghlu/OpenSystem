/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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

enum {
  Constant1,
  Constant2
};

enum {
  VarConstant1,
  VarConstant2
} global;

// NB: The order of fields here is important.
typedef struct Struct {
    int firstField;

    enum {
      NestedConstant1 = 0, NestedConstant2, NestedConstant3
    } adhocAnonEnumField;

    int lastField;
} Struct;

#if __OBJC__
enum : unsigned short {
  USConstant1,
  USConstant2
};

enum : unsigned short {
  USVarConstant1,
  USVarConstant2
} usGlobal;
#endif // __OBJC__
