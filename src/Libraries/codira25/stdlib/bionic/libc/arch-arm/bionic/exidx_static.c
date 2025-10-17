/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#include <link.h>

/* Find the .ARM.exidx section (which in the case of a static executable
 * can be identified through its start and end symbols), and return its
 * beginning and number of entries to the caller.  Note that for static
 * executables we do not need to use the value of the PC to find the
 * EXIDX section.
 */

struct exidx_entry {
  uint32_t key;
  uint32_t value;
};

extern struct exidx_entry __exidx_end;
extern struct exidx_entry __exidx_start;

_Unwind_Ptr dl_unwind_find_exidx(_Unwind_Ptr pc __attribute__((unused)), int* pcount) {
  *pcount = (&__exidx_end - &__exidx_start);
  return (_Unwind_Ptr)&__exidx_start;
}

_Unwind_Ptr __gnu_Unwind_Find_exidx(_Unwind_Ptr pc, int *pcount) {
  return dl_unwind_find_exidx(pc, pcount);
}
