/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 7, 2022.
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

#include "mymalloc.h"
#include "get_hostname.h"
#include "get_domainname.h"

/* Local stuff. */

static char *my_domain_name;

/* get_domainname - look up my domain name */

const char *get_domainname(void)
{
    const char *host;
    const char *dot;

    /*
     * Use the hostname when it is not a FQDN ("foo"), or when the hostname
     * actually is a domain name ("foo.com").
     */
    if (my_domain_name == 0) {
	host = get_hostname();
	if ((dot = strchr(host, '.')) == 0 || strchr(dot + 1, '.') == 0) {
	    my_domain_name = mystrdup(host);
	} else {
	    my_domain_name = mystrdup(dot + 1);
	}
    }
    return (my_domain_name);
}
