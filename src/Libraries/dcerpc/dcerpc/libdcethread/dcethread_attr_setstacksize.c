/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#include <config.h>
#include <limits.h>
#include <unistd.h>

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

int
dcethread_attr_setstacksize(dcethread_attr *attr, size_t stacksize)
{
    size_t new_stacksize = stacksize;
    int page_size;

    /* stack size can not be less than PTHREAD_STACK_MIN */
    if (new_stacksize < PTHREAD_STACK_MIN) {
        new_stacksize = PTHREAD_STACK_MIN;
    }

    /* stack size must be a multiple of system page size */
    page_size = getpagesize();
    if (new_stacksize % page_size) {
        new_stacksize = ((new_stacksize / page_size) + 1) * page_size;
    }

    return dcethread__set_errno(pthread_attr_setstacksize(attr, new_stacksize));
}

int
dcethread_attr_setstacksize_throw(dcethread_attr *attr, size_t stacksize)
{
    DCETHREAD_WRAP_THROW(dcethread_attr_setstacksize(attr, stacksize));
}
