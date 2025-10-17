/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
/* System library. */

#include <sys_defs.h>
#include <netdb.h>

/* Utility library. */

#include <vstring.h>

/* DNS library. */

#include "dns.h"

 /*
  * Mapping from error code to printable string. The herror() routine does
  * something similar, but has output only to the stderr stream.
  */
struct dns_error_map {
    unsigned error;
    const char *text;
};

static struct dns_error_map dns_error_map[] = {
    HOST_NOT_FOUND, "Host not found",
    TRY_AGAIN, "Host not found, try again",
    NO_RECOVERY, "Non-recoverable error",
    NO_DATA, "Host found but no data record of requested type",
};

/* dns_strerror - map resolver error code to printable string */

const char *dns_strerror(unsigned error)
{
    static VSTRING *unknown = 0;
    unsigned i;

    for (i = 0; i < sizeof(dns_error_map) / sizeof(dns_error_map[0]); i++)
	if (dns_error_map[i].error == error)
	    return (dns_error_map[i].text);
    if (unknown == 0)
	unknown = vstring_alloc(sizeof("Unknown error XXXXXX"));
    vstring_sprintf(unknown, "Unknown error %u", error);
    return (vstring_str(unknown));
}
