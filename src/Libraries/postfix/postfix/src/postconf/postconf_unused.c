/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#include <dict.h>
#include <vstream.h>

/* Global library. */

#include <mail_params.h>
#include <mail_conf.h>

/* Application-specific. */

#include <postconf.h>

/* pcf_flag_unused_parameters - warn about unused parameters */

static void pcf_flag_unused_parameters(DICT *dict, const char *conf_name,
				               PCF_MASTER_ENT *local_scope)
{
    const char *myname = "pcf_flag_unused_parameters";
    const char *param_name;
    const char *param_value;
    int     how;

    /*
     * Sanity checks.
     */
    if (pcf_param_table == 0)
	msg_panic("%s: global parameter table is not initialized", myname);

    /*
     * Iterate over all entries, and flag parameter names that aren't used
     * anywhere. Show the warning message at the end of the output.
     */
    if (dict->sequence == 0)
	msg_panic("%s: parameter dictionary %s has no iterator",
		  myname, conf_name);
    for (how = DICT_SEQ_FUN_FIRST;
	 dict->sequence(dict, how, &param_name, &param_value) == 0;
	 how = DICT_SEQ_FUN_NEXT) {
	if (PCF_PARAM_TABLE_LOCATE(pcf_param_table, param_name) == 0
	    && (local_scope == 0
		|| PCF_PARAM_TABLE_LOCATE(local_scope->valid_names, param_name) == 0)) {
	    vstream_fflush(VSTREAM_OUT);
	    msg_warn("%s/%s: unused parameter: %s=%s",
		     var_config_dir, conf_name, param_name, param_value);
	}
    }
}

/* pcf_flag_unused_main_parameters - warn about unused parameters */

void    pcf_flag_unused_main_parameters(void)
{
    const char *myname = "pcf_flag_unused_main_parameters";
    DICT   *dict;

    /*
     * Iterate over all main.cf entries, and flag parameter names that aren't
     * used anywhere.
     */
    if ((dict = dict_handle(CONFIG_DICT)) == 0)
	msg_panic("%s: parameter dictionary %s not found",
		  myname, CONFIG_DICT);
    pcf_flag_unused_parameters(dict, MAIN_CONF_FILE, (PCF_MASTER_ENT *) 0);
}

/* pcf_flag_unused_master_parameters - warn about unused parameters */

void    pcf_flag_unused_master_parameters(void)
{
    const char *myname = "pcf_flag_unused_master_parameters";
    PCF_MASTER_ENT *masterp;
    DICT   *dict;

    /*
     * Sanity checks.
     */
    if (pcf_master_table == 0)
	msg_panic("%s: master table is not initialized", myname);

    /*
     * Iterate over all master.cf entries, and flag parameter names that
     * aren't used anywhere.
     */
    for (masterp = pcf_master_table; masterp->argv != 0; masterp++)
	if ((dict = masterp->all_params) != 0)
	    pcf_flag_unused_parameters(dict, MASTER_CONF_FILE, masterp);
}
