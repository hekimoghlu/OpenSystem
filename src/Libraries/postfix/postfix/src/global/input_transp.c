/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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

#include <name_mask.h>
#include <msg.h>

/* Global library. */

#include <mail_params.h>
#include <cleanup_user.h>
#include <input_transp.h>

/* input_transp_mask - compute mail receive transparency mask */

int     input_transp_mask(const char *param_name, const char *pattern)
{
    static const NAME_MASK table[] = {
	"no_unknown_recipient_checks", INPUT_TRANSP_UNKNOWN_RCPT,
	"no_address_mappings", INPUT_TRANSP_ADDRESS_MAPPING,
	"no_header_body_checks", INPUT_TRANSP_HEADER_BODY,
	"no_milters", INPUT_TRANSP_MILTER,
	0,
    };

    return (name_mask(param_name, table, pattern));
}

/* input_transp_cleanup - adjust cleanup options */

int     input_transp_cleanup(int cleanup_flags, int transp_mask)
{
    const char *myname = "input_transp_cleanup";

    if (msg_verbose)
	msg_info("before %s: cleanup flags = %s",
		 myname, cleanup_strflags(cleanup_flags));
    if (transp_mask & INPUT_TRANSP_ADDRESS_MAPPING)
	cleanup_flags &= ~(CLEANUP_FLAG_BCC_OK | CLEANUP_FLAG_MAP_OK);
    if (transp_mask & INPUT_TRANSP_HEADER_BODY)
	cleanup_flags &= ~CLEANUP_FLAG_FILTER;
    if (transp_mask & INPUT_TRANSP_MILTER)
	cleanup_flags &= ~CLEANUP_FLAG_MILTER;
    if (msg_verbose)
	msg_info("after %s: cleanup flags = %s",
		 myname, cleanup_strflags(cleanup_flags));
    return (cleanup_flags);
}
