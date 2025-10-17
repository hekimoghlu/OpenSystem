/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#include <syslog.h>

#ifdef __APPLE__
#include <pthread.h>

static int SHOULD_DO_LOGGING(void)
{
    return pthread_is_threaded_np() && pthread_main_np();
}
#else
#define SHOULD_DO_LOGGING() (0)
#endif



static void
warn_blocking(void *ptr)
{
    syslog(LOG_NOTICE, "%s is called on main thread, its a blocking api", (const char *)ptr);
}

void
heim_warn_blocking(const char *apiname, heim_base_once_t *once)
{
    if (SHOULD_DO_LOGGING())
	heim_base_once_f(once, (void *)apiname, warn_blocking);
}
