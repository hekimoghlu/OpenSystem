/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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

// This header is included on non-ObjC platforms.

void overloaded(void) __attribute__((overloadable));
void overloaded(int) __attribute__((overloadable));

extern void use(const char *);

static inline void test_my_log() {
  __attribute__((internal_linkage)) static const char fmt[] = "foobar";
  use(fmt);
}

extern void useInt(unsigned int);

typedef struct {
    unsigned int val[8];
} a_thing;

static inline void log_a_thing(const a_thing thing) {
 useInt(thing.val[0]);
 useInt(thing.val[7]);
}

static inline unsigned int return7(void) {
  return 7;
}
