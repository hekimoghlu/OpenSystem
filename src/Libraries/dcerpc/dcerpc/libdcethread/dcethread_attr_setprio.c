/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#include <sched.h>
#ifdef HAVE_STRING_H
#    include <string.h>
#endif

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

int
dcethread_attr_setprio(dcethread_attr *attr, int priority)
{
    struct sched_param sp;

    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = priority;

    return dcethread__set_errno(pthread_attr_setschedparam(attr, &sp));
}

int
dcethread_attr_setprio_throw(dcethread_attr *attr, int priority)
{
    DCETHREAD_WRAP_THROW(dcethread_attr_setprio(attr, priority));
}
