/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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

/*
 *
 */

#ifdef __APPLE_PRIVATE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlanguage-extension-token"

const char *__crashreporter_info__ = NULL;
asm(".desc ___crashreporter_info__, 0x10");

#pragma clang diagnostic pop
#endif


/**
 * Abort and log the failure (using syslog)
 */

void
heim_abort(const char *fmt, ...)
    HEIMDAL_NORETURN_ATTRIBUTE
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 1, 2))
{
    va_list ap;
    va_start(ap, fmt);
    heim_abortv(fmt, ap);
    va_end(ap);
}

/**
 * Abort and log the failure (using syslog)
 */

void
heim_abortv(const char *fmt, va_list ap)
    HEIMDAL_NORETURN_ATTRIBUTE
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 1, 0))
{
    char *str = NULL;
    int ret;
    
    ret = vasprintf(&str, fmt, ap);
    if (ret > 0 && str) {
	syslog(LOG_ERR, "heim_abort: %s", str);
	
#ifdef __APPLE_PRIVATE__
	__crashreporter_info__ = str;
#endif
    }
    abort();
}
