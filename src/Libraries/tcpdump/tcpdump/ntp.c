/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "ntp.h"

#include "extract.h"

#define	JAN_1970	INT64_T_CONSTANT(2208988800)	/* 1970 - 1900 in seconds */

void
p_ntp_time(netdissect_options *ndo,
	   const struct l_fixedpt *lfp)
{
	uint32_t i;
	uint32_t uf;
	uint32_t f;
	double ff;

	i = GET_BE_U_4(lfp->int_part);
	uf = GET_BE_U_4(lfp->fraction);
	ff = uf;
	if (ff < 0.0)		/* some compilers are buggy */
		ff += FMAXINT;
	ff = ff / FMAXINT;			/* shift radix point by 32 bits */
	f = (uint32_t)(ff * 1000000000.0);	/* treat fraction as parts per billion */
	ND_PRINT("%u.%09u", i, f);

	/*
	 * print the UTC time in human-readable format.
	 */
	if (i) {
	    int64_t seconds_64bit = (int64_t)i - JAN_1970;
	    time_t seconds;
	    char time_buf[128];
	    const char *time_string;

	    seconds = (time_t)seconds_64bit;
	    if (seconds != seconds_64bit) {
		/*
		 * It doesn't fit into a time_t, so we can't hand it
		 * to gmtime.
		 */
		time_string = "[Time is too large to fit into a time_t]";
	    } else {
		/* use ISO 8601 (RFC3339) format */
		time_string = nd_format_time(time_buf, sizeof (time_buf),
		  "%Y-%m-%dT%H:%M:%SZ", gmtime(&seconds));
	    }
	    ND_PRINT(" (%s)", time_string);
	}
}
