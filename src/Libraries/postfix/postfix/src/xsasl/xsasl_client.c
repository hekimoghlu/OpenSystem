/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>

/* SASL implementations. */

#include <xsasl.h>
#include <xsasl_cyrus.h>

 /*
  * Lookup table for available SASL client implementations.
  */
typedef struct {
    char   *client_type;
    struct XSASL_CLIENT_IMPL *(*client_init) (const char *, const char *);
} XSASL_CLIENT_IMPL_INFO;

static const XSASL_CLIENT_IMPL_INFO client_impl_info[] = {
#ifdef XSASL_TYPE_CYRUS
    XSASL_TYPE_CYRUS, xsasl_cyrus_client_init,
#endif
    0,
};

/* xsasl_client_init - look up client implementation by name */

XSASL_CLIENT_IMPL *xsasl_client_init(const char *client_type,
				             const char *path_info)
{
    const XSASL_CLIENT_IMPL_INFO *xp;

    for (xp = client_impl_info; xp->client_type; xp++)
	if (strcmp(client_type, xp->client_type) == 0)
	    return (xp->client_init(client_type, path_info));
    msg_warn("unsupported SASL client implementation: %s", client_type);
    return (0);
}

/* xsasl_client_types - report available implementation types */

ARGV   *xsasl_client_types(void)
{
    const XSASL_CLIENT_IMPL_INFO *xp;
    ARGV   *argv = argv_alloc(1);

    for (xp = client_impl_info; xp->client_type; xp++)
	argv_add(argv, xp->client_type, ARGV_END);
    return (argv);
}
