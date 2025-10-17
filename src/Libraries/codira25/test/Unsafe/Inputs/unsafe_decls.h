/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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

void unsafe_c_function(void) __attribute__((language_attr("unsafe")));

struct __attribute__((language_attr("unsafe"))) UnsafeType {
  int field;
};

void print_ints(int *ptr, int count);

#define _CXX_INTEROP_STRINGIFY(_x) #_x

#define LANGUAGE_SHARED_REFERENCE(_retain, _release)                                \
  __attribute__((language_attr("import_reference")))                          \
  __attribute__((language_attr(_CXX_INTEROP_STRINGIFY(retain:_retain))))      \
  __attribute__((language_attr(_CXX_INTEROP_STRINGIFY(release:_release))))

#define LANGUAGE_SAFE __attribute__((language_attr("@safe")))
#define LANGUAGE_UNSAFE __attribute__((language_attr("@unsafe")))

struct NoPointers {
  float x, y, z;
};

union NoPointersUnion {
  float x;
  double y;
};

struct NoPointersUnsafe {
  float x, y, z;
} LANGUAGE_UNSAFE;

struct HasPointers {
  float *numbers;
};


union HasPointersUnion {
  float *numbers;
  double x;
};

struct HasPointersSafe {
  float *numbers;
} LANGUAGE_SAFE;

struct RefCountedType {
  void *ptr;
} LANGUAGE_SHARED_REFERENCE(RCRetain, RCRelease);

struct RefCountedType *RCRetain(struct RefCountedType *object);
void RCRelease(struct RefCountedType *object);

struct HasRefCounted {
  struct RefCountedType *ref;
};

struct ListNode {
  double data;
  struct ListNode *next;
};

enum SomeColor {
  SCRed,
  SCGreen,
  SCBlue
};
