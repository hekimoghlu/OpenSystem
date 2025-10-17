/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

#include <msg.h>
#include <mymalloc.h>
#include <vstring.h>

/* Application-specific. */

#include <postconf.h>

VSTRING *pcf_param_string_buf;

/* pcf_make_param_node - make node for global parameter table */

PCF_PARAM_NODE *pcf_make_param_node(int flags, void *param_data,
				         const char *(*convert_fn) (void *))
{
    PCF_PARAM_NODE *node;

    node = (PCF_PARAM_NODE *) mymalloc(sizeof(*node));
    node->flags = flags;
    node->param_data = param_data;
    node->convert_fn = convert_fn;
    return (node);
}

/* pcf_convert_param_node - get default parameter value */

const char *pcf_convert_param_node(int mode, const char *name, PCF_PARAM_NODE *node)
{
    const char *myname = "pcf_convert_param_node";
    const char *value;

    /*
     * One-off initialization.
     */
    if (pcf_param_string_buf == 0)
	pcf_param_string_buf = vstring_alloc(100);

    /*
     * Sanity check. A null value indicates that a parameter does not have
     * the requested value. At this time, the only requested value can be the
     * default value, and a null pointer value makes no sense here.
     */
    if ((mode & PCF_SHOW_DEFS) == 0)
	msg_panic("%s: request for non-default value of parameter %s",
		  myname, name);
    if ((value = node->convert_fn(node->param_data)) == 0)
	msg_panic("%s: parameter %s has null pointer default value",
		  myname, name);

    /*
     * Return the parameter default value.
     */
    return (value);
}
