/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
#include <xsasl_dovecot.h>

 /*
  * Lookup table for available SASL server implementations.
  */
typedef struct {
    char   *server_type;
    struct XSASL_SERVER_IMPL *(*server_init) (const char *, const char *);
} XSASL_SERVER_IMPL_INFO;

static const XSASL_SERVER_IMPL_INFO server_impl_info[] = {
#ifdef XSASL_TYPE_CYRUS
    {XSASL_TYPE_CYRUS, xsasl_cyrus_server_init},
#endif
#ifdef XSASL_TYPE_DOVECOT
    {XSASL_TYPE_DOVECOT, xsasl_dovecot_server_init},
#endif
    {0, 0}
};

/* xsasl_server_init - look up server implementation by name */

XSASL_SERVER_IMPL *xsasl_server_init(const char *server_type,
				             const char *path_info)
{
    const XSASL_SERVER_IMPL_INFO *xp;

    for (xp = server_impl_info; xp->server_type; xp++)
	if (strcmp(server_type, xp->server_type) == 0)
	    return (xp->server_init(server_type, path_info));
    msg_warn("unsupported SASL server implementation: %s", server_type);
    return (0);
}

/* xsasl_server_types - report available implementation types */

ARGV   *xsasl_server_types(void)
{
    const XSASL_SERVER_IMPL_INFO *xp;
    ARGV   *argv = argv_alloc(1);

    for (xp = server_impl_info; xp->server_type; xp++)
	argv_add(argv, xp->server_type, ARGV_END);
    return (argv);
}
