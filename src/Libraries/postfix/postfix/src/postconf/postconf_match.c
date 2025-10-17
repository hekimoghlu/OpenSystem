/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#include <vstring.h>

/* Global library. */

#include <split_at.h>

/* Application-specific. */

#include <postconf.h>

 /*
  * Conversion table. Each PCF_MASTER_NAME_XXX name entry must be stored at
  * table offset PCF_MASTER_FLD_XXX. So don't mess it up.
  */
NAME_CODE pcf_field_name_offset[] = {
    PCF_MASTER_NAME_SERVICE, PCF_MASTER_FLD_SERVICE,
    PCF_MASTER_NAME_TYPE, PCF_MASTER_FLD_TYPE,
    PCF_MASTER_NAME_PRIVATE, PCF_MASTER_FLD_PRIVATE,
    PCF_MASTER_NAME_UNPRIV, PCF_MASTER_FLD_UNPRIV,
    PCF_MASTER_NAME_CHROOT, PCF_MASTER_FLD_CHROOT,
    PCF_MASTER_NAME_WAKEUP, PCF_MASTER_FLD_WAKEUP,
    PCF_MASTER_NAME_MAXPROC, PCF_MASTER_FLD_MAXPROC,
    PCF_MASTER_NAME_CMD, PCF_MASTER_FLD_CMD,
    "*", PCF_MASTER_FLD_WILDC,
    0, PCF_MASTER_FLD_NONE,
};

/* pcf_parse_field_pattern - parse service attribute pattern */

int     pcf_parse_field_pattern(const char *field_name)
{
    int     field_pattern;

    if ((field_pattern = name_code(pcf_field_name_offset,
				   NAME_CODE_FLAG_STRICT_CASE,
				   field_name)) == PCF_MASTER_FLD_NONE)
	msg_fatal("invalid service attribute name: \"%s\"", field_name);
    return (field_pattern);
}

/* pcf_parse_service_pattern - parse service pattern */

ARGV   *pcf_parse_service_pattern(const char *pattern, int min_expr, int max_expr)
{
    ARGV   *argv;
    char  **cpp;

    /*
     * Work around argv_split() lameness.
     */
    if (*pattern == '/')
	return (0);
    argv = argv_split(pattern, PCF_NAMESP_SEP_STR);
    if (argv->argc < min_expr || argv->argc > max_expr) {
	argv_free(argv);
	return (0);
    }

    /*
     * Allow '*' only all by itself.
     */
    for (cpp = argv->argv; *cpp; cpp++) {
	if (!PCF_MATCH_ANY(*cpp) && strchr(*cpp, PCF_MATCH_WILDC_STR[0]) != 0) {
	    argv_free(argv);
	    return (0);
	}
    }

    /*
     * Provide defaults for missing fields.
     */
    while (argv->argc < max_expr)
	argv_add(argv, PCF_MATCH_WILDC_STR, ARGV_END);
    return (argv);
}
