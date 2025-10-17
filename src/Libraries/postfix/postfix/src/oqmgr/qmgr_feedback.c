/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#include <stdlib.h>
#include <limits.h>			/* INT_MAX */
#include <stdio.h>			/* sscanf() */
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <name_code.h>
#include <stringops.h>
#include <mymalloc.h>

/* Global library. */

#include <mail_params.h>
#include <mail_conf.h>

/* Application-specific. */

#include "qmgr.h"

 /*
  * Lookup tables for main.cf feedback method names.
  */
const NAME_CODE qmgr_feedback_map[] = {
    CONC_FDBACK_NAME_WIN, QMGR_FEEDBACK_IDX_WIN,
#ifdef QMGR_FEEDBACK_IDX_SQRT_WIN
    CONC_FDBACK_NAME_SQRT_WIN, QMGR_FEEDBACK_IDX_SQRT_WIN,
#endif
    0, QMGR_FEEDBACK_IDX_NONE,
};

/* qmgr_feedback_init - initialize feedback control */

void    qmgr_feedback_init(QMGR_FEEDBACK *fb,
			           const char *name_prefix,
			           const char *name_tail,
			           const char *def_name,
			           const char *def_val)
{
    double  enum_val;
    char    denom_str[30 + 1];
    double  denom_val;
    char    slash;
    char    junk;
    char   *fbck_name;
    char   *fbck_val;

    /*
     * Look up the transport-dependent feedback value.
     */
    fbck_name = concatenate(name_prefix, name_tail, (char *) 0);
    fbck_val = get_mail_conf_str(fbck_name, def_val, 1, 0);

    /*
     * We allow users to express feedback as 1/8, as a more user-friendly
     * alternative to 0.125 (or worse, having users specify the number of
     * events in a feedback hysteresis cycle).
     * 
     * We use some sscanf() fu to parse the value into numerator and optional
     * "/" followed by denominator. We're doing this only a few times during
     * the process life time, so we strive for convenience instead of speed.
     */
#define INCLUSIVE_BOUNDS(val, low, high) ((val) >= (low) && (val) <= (high))

    fb->hysteresis = 1;				/* legacy */
    fb->base = -1;				/* assume error */

    switch (sscanf(fbck_val, "%lf %1[/] %30s%c",
		   &enum_val, &slash, denom_str, &junk)) {
    case 1:
	fb->index = QMGR_FEEDBACK_IDX_NONE;
	fb->base = enum_val;
	break;
    case 3:
	if ((fb->index = name_code(qmgr_feedback_map, NAME_CODE_FLAG_NONE,
				   denom_str)) != QMGR_FEEDBACK_IDX_NONE) {
	    fb->base = enum_val;
	} else if (INCLUSIVE_BOUNDS(enum_val, 0, INT_MAX)
		   && sscanf(denom_str, "%lf%c", &denom_val, &junk) == 1
		   && INCLUSIVE_BOUNDS(denom_val, 1.0 / INT_MAX, INT_MAX)) {
	    fb->base = enum_val / denom_val;
	}
	break;
    }

    /*
     * Sanity check. If input is bad, we just warn and use a reasonable
     * default.
     */
    if (!INCLUSIVE_BOUNDS(fb->base, 0, 1)) {
	msg_warn("%s: ignoring malformed or unreasonable feedback: %s",
		 strcmp(fbck_val, def_val) ? fbck_name : def_name, fbck_val);
	fb->index = QMGR_FEEDBACK_IDX_NONE;
	fb->base = 1;
    }

    /*
     * Performance debugging/analysis.
     */
    if (var_conc_feedback_debug)
	msg_info("%s: %s feedback type %d value at %d: %g",
		 name_prefix, strcmp(fbck_val, def_val) ?
		 fbck_name : def_name, fb->index, var_init_dest_concurrency,
		 QMGR_FEEDBACK_VAL(*fb, var_init_dest_concurrency));

    myfree(fbck_name);
    myfree(fbck_val);
}
