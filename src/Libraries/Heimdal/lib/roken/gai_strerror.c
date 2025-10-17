/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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

#include "roken.h"

static struct gai_error {
    int code;
    const char *str;
} errors[] = {
{EAI_NOERROR,		"no error"},
#ifdef EAI_ADDRFAMILY
{EAI_ADDRFAMILY,	"address family for nodename not supported"},
#endif
{EAI_AGAIN,		"temporary failure in name resolution"},
{EAI_BADFLAGS,		"invalid value for ai_flags"},
{EAI_FAIL,		"non-recoverable failure in name resolution"},
{EAI_FAMILY,		"ai_family not supported"},
{EAI_MEMORY,		"memory allocation failure"},
#ifdef EAI_NODATA
{EAI_NODATA,		"no address associated with nodename"},
#endif
{EAI_NONAME,		"nodename nor servname provided, or not known"},
{EAI_SERVICE,		"servname not supported for ai_socktype"},
{EAI_SOCKTYPE,		"ai_socktype not supported"},
{EAI_SYSTEM,		"system error returned in errno"},
{0,			NULL},
};

/*
 *
 */

ROKEN_LIB_FUNCTION const char * ROKEN_LIB_CALL
gai_strerror(int ecode)
{
    struct gai_error *g;

    for (g = errors; g->str != NULL; ++g)
	if (g->code == ecode)
	    return g->str;
    return "unknown error code in gai_strerror";
}
