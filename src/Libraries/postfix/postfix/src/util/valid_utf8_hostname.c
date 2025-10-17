/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
/*
  * System library.
  */
#include <sys_defs.h>

 /*
  * Utility library.
  */
#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>
#include <valid_hostname.h>
#include <midna_domain.h>
#include <valid_utf8_hostname.h>

/* valid_utf8_hostname - validate internationalized domain name */

int     valid_utf8_hostname(int enable_utf8, const char *name, int gripe)
{
    static const char myname[] = "valid_utf8_hostname";

    /*
     * Trivial cases first.
     */
    if (*name == 0) {
	if (gripe)
	    msg_warn("%s: empty domain name", myname);
	return (0);
    }

    /*
     * Convert non-ASCII domain name to ASCII and validate the result per
     * STD3. midna_domain_to_ascii() applies valid_hostname() to the result.
     * Propagate the gripe parameter for better diagnostics (note that
     * midna_domain_to_ascii() logs a problem only when the result is not
     * cached).
     */
#ifndef NO_EAI
    if (enable_utf8 && !allascii(name)) {
	if (midna_domain_to_ascii(name) == 0) {
	    if (gripe)
		msg_warn("%s: malformed UTF-8 domain name", myname);
	    return (0);
	} else {
	    return (1);
	}
    }
#endif

    /*
     * Validate ASCII name per STD3.
     */
    return (valid_hostname(name, gripe));
}
