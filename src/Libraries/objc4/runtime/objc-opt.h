/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
/*
  objc-opt.h
  Management of optimizations in the dyld shared cache
*/

#ifndef _OBJC_OPT_H
#define _OBJC_OPT_H

#include <stdint.h>

typedef struct header_info_rw {

    bool getLoaded() const {
        return isLoaded;
    }

    void setLoaded(bool v) {
        isLoaded = v ? 1: 0;
    }

    struct header_info *getNext() const {
        return (struct header_info *)(next << 2);
    }

    void setNext(struct header_info *v) {
        next = ((uintptr_t)v) >> 2;
    }

private:
#ifdef __LP64__
    uintptr_t isLoaded                : 1;
    [[maybe_unused]] uintptr_t unused : 1;
    uintptr_t next                    : 62;
#else
    uintptr_t isLoaded                : 1;
    [[maybe_unused]] uintptr_t unused : 1;
    uintptr_t next                    : 30;
#endif
} header_info_rw;

struct objc_headeropt_rw_t {
    uint32_t count;
    uint32_t entsize;
    header_info_rw headers[0];  // sorted by mhdr address
};

#endif // _OBJC_OPT_H
