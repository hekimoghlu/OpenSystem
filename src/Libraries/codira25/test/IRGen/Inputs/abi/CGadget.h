/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

struct MyRect {
  float x;
  float y;
  float width;
  float height;
};

struct Trio {
  double i;
  double j;
  double k;
};

struct IntPair {
  int a;
  int b;
};

struct NestedInts {
  struct A {
    int value;
  } a;
  struct B {
    int value;
  } b;
};

struct BigStruct {
  char a[32];
};

enum RawEnum {
  Intergalactic,
  Planetary
};

typedef struct One {
  float first;
  float second;
} One;

static inline One makeOne(float f, float s) {
  One one;
  one.first = f;
  one.second = s;

  return one;
}

static inline float MyRect_Area(struct MyRect rect) {
  return rect.width * rect.height;
}
