/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#include "baselocl.h"

heim_queue_t
heim_queue_create(const char *name, heim_queue_attr_t attr)
{
    abort();
    return NULL;
}

void
heim_async_f(heim_queue_t queue, void *ctx, void (*callback)(void *data))
{
    abort();
}

void
heim_queue_release(heim_queue_t queue)
{
    abort();
}

heim_sema_t
heim_sema_create(long count)
{
    abort();
    return NULL;
}

void
heim_sema_signal(heim_sema_t sema)
{
    abort();
}

void
heim_sema_wait(heim_sema_t sema, time_t t)
{
    abort();
}
