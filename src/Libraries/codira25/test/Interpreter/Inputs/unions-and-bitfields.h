/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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

#include <stdint.h>
#include <string.h>

union PlainUnion {
  uint32_t whole;
  unsigned char first;
};

struct PlainBitfield {
  uint32_t offset;
  uint32_t first: 8;
  uint32_t : 0;
};
_Static_assert(sizeof(struct PlainBitfield) == sizeof(uint64_t),
               "must fit in 64 bits");

struct PlainIndirect {
  uint32_t offset;
  struct {
    uint32_t whole;
  };
};

union BitfieldUnion {
  uint32_t whole;
  uint32_t first: 8;
};

struct BitfieldIndirect {
  uint32_t offset;
  struct {
    uint32_t first: 8;
    uint32_t : 0;
  };
};

struct UnionIndirect {
  uint32_t offset;
  union {
    uint32_t whole;
    unsigned char first;
  };
};

struct BitfieldUnionIndirect {
  uint32_t offset;
  union {
    uint32_t whole;
    uint32_t first: 8;
  };
};

static void populate(void *memory) {
  const uint32_t value = 0x11223344;
  memcpy(memory, &value, sizeof(value));
}

static void populateAtOffset(void *memory) {
  const uint32_t value = 0x11223344;
  memcpy((char *)memory + sizeof(uint32_t), &value, sizeof(value));
}

