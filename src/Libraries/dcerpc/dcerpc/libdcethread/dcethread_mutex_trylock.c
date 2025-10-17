/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#include <errno.h>

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

int
dcethread_mutex_trylock(dcethread_mutex *mutex)
{
    int ret;

    ret = pthread_mutex_trylock((pthread_mutex_t*) &mutex->mutex);

    if (ret == 0)
    {
        mutex->owner = pthread_self();
	return 1;
    }
    else if (ret == EBUSY)
    {
	return 0;
    }
    else
    {
	return dcethread__set_errno(ret);
    }
}

int
dcethread_mutex_trylock_throw(dcethread_mutex *mutex)
{
    DCETHREAD_WRAP_THROW(dcethread_mutex_trylock(mutex));
}
