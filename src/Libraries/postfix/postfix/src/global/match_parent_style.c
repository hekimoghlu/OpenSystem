/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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

/* Utility library. */

/* Global library. */

#include <string_list.h>
#include <mail_params.h>
#include <match_parent_style.h>

/* Application-specific. */

static STRING_LIST *match_par_dom_list;

int     match_parent_style(const char *name)
{
    int     result;

    /*
     * Initialize on the fly.
     */
    if (match_par_dom_list == 0)
	match_par_dom_list =
	    string_list_init(VAR_PAR_DOM_MATCH, MATCH_FLAG_NONE,
			     var_par_dom_match);

    /*
     * Look up the parent domain matching policy.
     */
    if (string_list_match(match_par_dom_list, name))
	result = MATCH_FLAG_PARENT;
    else
	result = 0;
    return (result);
}
