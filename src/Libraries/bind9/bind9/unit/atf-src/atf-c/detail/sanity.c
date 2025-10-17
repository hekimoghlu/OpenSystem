/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

#include <err.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "sanity.h"

static
void
fail(const char *fmt, ...)
{
    va_list ap;
    char buf[4096];

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    warnx("%s", buf);
    warnx("%s", "");
    warnx("This is probably a bug in this application or one of the "
          "libraries it uses.  If you believe this problem is caused "
          "by, or is related to " PACKAGE_STRING ", please report it "
          "to " PACKAGE_BUGREPORT " and provide as many detatils as "
          "possible describing how you got to this condition.");

    abort();
}

void
atf_sanity_inv(const char *file, int line, const char *cond)
{
    fail("Invariant check failed at %s:%d: %s", file, line, cond);
}

void
atf_sanity_pre(const char *file, int line, const char *cond)
{
    fail("Precondition check failed at %s:%d: %s", file, line, cond);
}

void
atf_sanity_post(const char *file, int line, const char *cond)
{
    fail("Postcondition check failed at %s:%d: %s", file, line, cond);
}
