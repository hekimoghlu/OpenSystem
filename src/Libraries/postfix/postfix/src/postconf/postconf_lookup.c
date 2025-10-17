/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
#include <dict.h>
#include <stringops.h>
#include <mac_expand.h>

/* Global library. */

#include <mail_conf.h>

/* Application-specific. */

#include <postconf.h>

#define STR(x) vstring_str(x)

/* pcf_lookup_parameter_value - look up specific parameter value */

const char *pcf_lookup_parameter_value(int mode, const char *name,
				               PCF_MASTER_ENT *local_scope,
				               PCF_PARAM_NODE *node)
{
    const char *value = 0;

    /*
     * Local name=value entries in master.cf take precedence over global
     * name=value entries in main.cf. Built-in defaults have the lowest
     * precedence.
     */
    if ((mode & PCF_SHOW_DEFS) != 0
	|| ((local_scope == 0 || local_scope->all_params == 0
	     || (value = dict_get(local_scope->all_params, name)) == 0)
	    && (value = dict_lookup(CONFIG_DICT, name)) == 0
	    && (mode & PCF_SHOW_NONDEF) == 0)) {
	if (node != 0 || (node = PCF_PARAM_TABLE_FIND(pcf_param_table, name)) != 0)
	    value = pcf_convert_param_node(PCF_SHOW_DEFS, name, node);
    }
    return (value);
}

 /*
  * Data structure to pass private state while recursively expanding $name in
  * parameter values.
  */
typedef struct {
    int     mode;
    PCF_MASTER_ENT *local_scope;
} PCF_EVAL_CTX;

/* pcf_lookup_parameter_value_wrapper - macro parser call-back routine */

static const char *pcf_lookup_parameter_value_wrapper(const char *key,
						            int unused_type,
						              void *context)
{
    PCF_EVAL_CTX *cp = (PCF_EVAL_CTX *) context;

    return (pcf_lookup_parameter_value(cp->mode, key, cp->local_scope,
				       (PCF_PARAM_NODE *) 0));
}

/* pcf_expand_parameter_value - expand $name in parameter value */

char   *pcf_expand_parameter_value(VSTRING *buf, int mode, const char *value,
				           PCF_MASTER_ENT *local_scope)
{
    const char *myname = "pcf_expand_parameter_value";
    static VSTRING *local_buf;
    int     status;
    PCF_EVAL_CTX eval_ctx;

    /*
     * Initialize.
     */
    if (buf == 0) {
	if (local_buf == 0)
	    local_buf = vstring_alloc(10);
	buf = local_buf;
    }

    /*
     * Expand macros recursively.
     * 
     * When expanding $name in "postconf -n" parameter values, don't limit the
     * search to only non-default parameter values.
     * 
     * When expanding $name in "postconf -d" parameter values, do limit the
     * search to only default parameter values.
     */
#define DONT_FILTER (char *) 0

    eval_ctx.mode = (mode & ~PCF_SHOW_NONDEF);
    eval_ctx.local_scope = local_scope;
    status = mac_expand(buf, value, MAC_EXP_FLAG_RECURSE, DONT_FILTER,
		    pcf_lookup_parameter_value_wrapper, (void *) &eval_ctx);
    if (status & MAC_PARSE_ERROR)
	msg_fatal("macro processing error");
    if (msg_verbose > 1) {
	if (strcmp(value, STR(buf)) != 0)
	    msg_info("%s: expand %s -> %s", myname, value, STR(buf));
	else
	    msg_info("%s: const  %s", myname, value);
    }
    return (STR(buf));
}
