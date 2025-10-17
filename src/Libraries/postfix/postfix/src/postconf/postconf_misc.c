/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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

#include <mymalloc.h>
#include <safe.h>

/* Global library. */

#include <mail_params.h>
#include <mail_conf.h>

/* Application-specific. */

#include <postconf.h>

/* pcf_set_config_dir - forcibly override var_config_dir */

void    pcf_set_config_dir(void)
{
    char   *config_dir;

    if (var_config_dir)
	myfree(var_config_dir);
    if ((config_dir = safe_getenv(CONF_ENV_PATH)) != 0) {
	var_config_dir = mystrdup(config_dir);
	set_mail_conf_str(VAR_CONFIG_DIR, var_config_dir);
    } else {
	var_config_dir = mystrdup(DEF_CONFIG_DIR);
    }
}
