/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

int
dcethread_attr_create(dcethread_attr *attr)
{
    if (dcethread__set_errno(pthread_attr_init(attr)))
    {
	return -1;
    }

    if (dcethread__set_errno(pthread_attr_setdetachstate(attr, PTHREAD_CREATE_JOINABLE)))
    {
	pthread_attr_destroy(attr);
	return -1;
    }

    return 0;
}

int
dcethread_attr_create_throw(dcethread_attr *attr)
{
    DCETHREAD_WRAP_THROW(dcethread_attr_create(attr));
}
