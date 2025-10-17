/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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

#ifndef HAVE_H_ERRNO
static int h_errno = NO_RECOVERY;
#endif

/*
 * lookup `name' (address family `af') in DNS and return a pointer
 * to a malloced struct hostent or NULL.
 */

ROKEN_LIB_FUNCTION struct hostent * ROKEN_LIB_CALL
getipnodebyname (const char *name, int af, int flags, int *error_num)
{
    struct hostent *tmp;

#ifdef HAVE_GETHOSTBYNAME2
    tmp = gethostbyname2 (name, af);
#else
    if (af != AF_INET) {
	*error_num = NO_ADDRESS;
	return NULL;
    }
    tmp = gethostbyname (name);
#endif
    if (tmp == NULL) {
	switch (h_errno) {
	case HOST_NOT_FOUND :
	case TRY_AGAIN :
	case NO_RECOVERY :
	    *error_num = h_errno;
	    break;
	case NO_DATA :
	    *error_num = NO_ADDRESS;
	    break;
	default :
	    *error_num = NO_RECOVERY;
	    break;
	}
	return NULL;
    }
    tmp = copyhostent (tmp);
    if (tmp == NULL) {
	*error_num = TRY_AGAIN;
	return NULL;
    }
    return tmp;
}
