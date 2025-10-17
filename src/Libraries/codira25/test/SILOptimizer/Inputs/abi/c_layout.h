/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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

struct BitfieldOne {
  unsigned a;
  unsigned : 0;
  struct Nested {
    float x;
    unsigned y : 15;
    unsigned z : 8;
  } b;
  int c : 5;
  int d : 7;
  int e : 13;
  int f : 15;
  int g : 8;
  int h : 2;
  float i;
  int j : 3;
  int k : 4;
  unsigned long long l;
  unsigned m;
};

struct BitfieldOne createBitfieldOne(void);
void consumeBitfieldOne(struct BitfieldOne one);

