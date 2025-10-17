/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#if defined(HAVE_CONFIG_H)
#include "bconfig.h"
#endif

#include <errno.h>
#include <stdlib.h>

#include "atf-c/error.h"

#include "env.h"
#include "sanity.h"
#include "text.h"

const char *
atf_env_get(const char *name)
{
    const char* val = getenv(name);
    PRE(val != NULL);
    return val;
}

bool
atf_env_has(const char *name)
{
    return getenv(name) != NULL;
}

atf_error_t
atf_env_set(const char *name, const char *val)
{
    atf_error_t err;

#if defined(HAVE_SETENV)
    if (setenv(name, val, 1) == -1)
        err = atf_libc_error(errno, "Cannot set environment variable "
                             "'%s' to '%s'", name, val);
    else
        err = atf_no_error();
#elif defined(HAVE_PUTENV)
    char *buf;

    err = atf_text_format(&buf, "%s=%s", name, val);
    if (!atf_is_error(err)) {
        if (putenv(buf) == -1)
            err = atf_libc_error(errno, "Cannot set environment variable "
                                 "'%s' to '%s'", name, val);
        free(buf);
    }
#else
#   error "Don't know how to set an environment variable."
#endif

    return err;
}

atf_error_t
atf_env_unset(const char *name)
{
    atf_error_t err;

#if defined(HAVE_UNSETENV)
    unsetenv(name);
    err = atf_no_error();
#elif defined(HAVE_PUTENV)
    char *buf;

    err = atf_text_format(&buf, "%s=", name);
    if (!atf_is_error(err)) {
        if (putenv(buf) == -1)
            err = atf_libc_error(errno, "Cannot unset environment variable"
                                 " '%s'", name);
        free(buf);
    }
#else
#   error "Don't know how to unset an environment variable."
#endif

    return err;
}
