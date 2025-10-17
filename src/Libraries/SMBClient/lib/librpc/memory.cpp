/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
#include "memory.hpp"
#include "compiler.h"

#include <cstdlib>
#include <new>

namespace platform {

void
invoke_new_handler(void)
{
    std::new_handler handler;

    // Get the curent new_handler by double-swapping. If multiple threads
    // race over this, then we will end up aborting. Ce la vie.
    handler = std::set_new_handler(::abort);
    std::set_new_handler(handler);

    handler();
}

void *
allocate(
        void * buf,
        std::size_t nbytes)
{
retry:
    if (buf == NULL) {
        if (nbytes % platform::pagesize()) {
            buf = ::malloc(nbytes);
        } else {
            buf = ::valloc(nbytes);
        }
    } else {
        buf = ::realloc(buf, nbytes);
    }

    if (UNLIKELY(buf == NULL)) {
        invoke_new_handler();
        goto retry;
    }

    return buf;
}

} // namespace platform
/* vim: set ts=4 sw=4 tw=79 et cindent : */
