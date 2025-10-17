/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
/* System library.*/

#include <sys_defs.h>

/* Utility library. */

#include <name_mask.h>

/* Global library. */

#include <ehlo_mask.h>

 /*
  * The lookup table.
  */
static const NAME_MASK ehlo_mask_table[] = {
    "8BITMIME", EHLO_MASK_8BITMIME,
    "AUTH", EHLO_MASK_AUTH,
    "ETRN", EHLO_MASK_ETRN,
    "PIPELINING", EHLO_MASK_PIPELINING,
    "SIZE", EHLO_MASK_SIZE,
    "VERP", EHLO_MASK_VERP,
    "VRFY", EHLO_MASK_VRFY,
    "XCLIENT", EHLO_MASK_XCLIENT,
    "XFORWARD", EHLO_MASK_XFORWARD,
    "STARTTLS", EHLO_MASK_STARTTLS,
    "ENHANCEDSTATUSCODES", EHLO_MASK_ENHANCEDSTATUSCODES,
    "DSN", EHLO_MASK_DSN,
    "EHLO_MASK_SMTPUTF8", EHLO_MASK_SMTPUTF8,
    "SILENT-DISCARD", EHLO_MASK_SILENT,	/* XXX In-band signaling */
    0,
};

/* ehlo_mask - string to bit mask */

int     ehlo_mask(const char *mask_str)
{

    /*
     * We allow "STARTTLS" besides "starttls, because EHLO keywords are often
     * spelled in uppercase. We ignore non-existent EHLO keywords so people
     * can switch between Postfix versions without trouble.
     */
    return (name_mask_opt("ehlo string mask", ehlo_mask_table,
			  mask_str, NAME_MASK_ANY_CASE | NAME_MASK_IGNORE));
}

/* str_ehlo_mask - mask to string */

const char *str_ehlo_mask(int mask_bits)
{

    /*
     * We don't allow non-existent bits. Doing so makes no sense at this
     * time.
     */
    return (str_name_mask("ehlo bitmask", ehlo_mask_table, mask_bits));
}

#ifdef TEST

 /*
  * Stand-alone test program.
  */
#include <stdlib.h>
#include <vstream.h>
#include <vstring.h>
#include <vstring_vstream.h>

int     main(int unused_argc, char **unused_argv)
{
    int     mask_bits;
    VSTRING *buf = vstring_alloc(1);
    const char *mask_string;

    while (vstring_get_nonl(buf, VSTREAM_IN) != VSTREAM_EOF) {
	mask_bits = ehlo_mask(vstring_str(buf));
	mask_string = str_ehlo_mask(mask_bits);
	vstream_printf("%s -> 0x%x -> %s\n", vstring_str(buf), mask_bits,
		       mask_string);
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(buf);
    exit(0);
}

#endif
