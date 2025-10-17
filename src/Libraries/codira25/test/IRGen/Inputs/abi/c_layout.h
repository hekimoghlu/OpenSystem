/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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

struct A {
  int x;
};
struct A createA(void);

enum CrappyColor {
  Red, Green, Blue
};

struct BitfieldSeparatorReference {
  unsigned char a;
  unsigned : 0;
  unsigned char b;
};

typedef struct BitfieldSeparatorSameName {
  unsigned char a;
  unsigned : 0;
  unsigned char b;
} BitfieldSeparatorSameName;

typedef struct BitfieldSeparatorDifferentNameStruct {
  unsigned char a;
  unsigned : 0;
  unsigned char b;
} BitfieldSeparatorDifferentName;

typedef struct {
  unsigned char a;
  unsigned : 0;
  unsigned char b;
} BitfieldSeparatorAnon;

typedef float vector_float3 __attribute__((__ext_vector_type__(3)));

struct SIMDStruct {
  vector_float3 v;
};

void takesSIMDStruct(struct SIMDStruct);

struct HasRecursivePointers {
  struct HasRecursivePointers *next;
  void (*getNext)(struct HasRecursivePointers);
};

struct HasNestedUnion {
  struct {
    int x;
    float f;
  } s;
};

// Test sign extension behavior

char chareth(char a);
signed char signedChareth(signed char a);
unsigned char unsignedChareth(unsigned char a);

short eatMyShorts(short a);
unsigned short eatMyUnsignedShorts(unsigned short a);

int ints(int a);
unsigned unsigneds(unsigned a);

// Test static globals

static float staticFloat = 17.0;
static int staticInt = 42;
static const char * const staticString = "abc";

static inline void doubleTrouble(void) {
  staticFloat *= 2.0;
}

#define LANGUAGE_ENUM(_type, _name) enum _name

typedef LANGUAGE_ENUM(int, AmazingColor) {
  Cyan, Magenta, Yellow
} AmazingColor;
