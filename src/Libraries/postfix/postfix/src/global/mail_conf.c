/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstream.h>
#include <vstring.h>
#include <dict.h>
#include <safe.h>
#include <stringops.h>
#include <readlline.h>

/* Global library. */

#include "mail_params.h"
#include "mail_conf.h"

/* mail_conf_checkdir - authorize non-default directory */

void mail_conf_checkdir(const char *config_dir)
{
    VSTRING *buf;
    VSTREAM *fp;
    char   *path;
    char   *name;
    char   *value;
    char   *cp;
    int     found = 0;

    /*
     * If running set-[ug]id, require that a non-default configuration
     * directory name is blessed as a bona fide configuration directory in
     * the default main.cf file.
     */
    path = concatenate(DEF_CONFIG_DIR, "/", "main.cf", (char *) 0);
    if ((fp = vstream_fopen(path, O_RDONLY, 0)) == 0)
	msg_fatal("open file %s: %m", path);

    buf = vstring_alloc(1);
    while (found == 0 && readlline(buf, fp, (int *) 0)) {
	if (split_nameval(vstring_str(buf), &name, &value) == 0
	    && (strcmp(name, VAR_CONFIG_DIRS) == 0
		|| strcmp(name, VAR_MULTI_CONF_DIRS) == 0)) {
	    while (found == 0 && (cp = mystrtok(&value, CHARS_COMMA_SP)) != 0)
		if (strcmp(cp, config_dir) == 0)
		    found = 1;
	}
    }
    if (vstream_fclose(fp))
	msg_fatal("read file %s: %m", path);
    vstring_free(buf);

    if (found == 0) {
	msg_error("unauthorized configuration directory name: %s", config_dir);
	msg_fatal("specify \"%s = %s\" or \"%s = %s\" in %s",
		  VAR_CONFIG_DIRS, config_dir,
		  VAR_MULTI_CONF_DIRS, config_dir, path);
    }
    myfree(path);
}

/* mail_conf_read - read global configuration file */

void    mail_conf_read(void)
{
    mail_conf_suck();
    mail_params_init();
}

/* mail_conf_suck - suck in the global configuration file */

void    mail_conf_suck(void)
{
    char   *config_dir;
    char   *path;

    /*
     * The code below requires that all configuration directory override
     * mechanisms set the CONF_ENV_PATH environment variable, even if the
     * override was specified via the command line. This reduces the number
     * of pathways that need to be checked for possible security attacks.
     * 
     * Note: this code necessarily runs before cleanenv() can enforce the
     * import_environment scrubbing policy.
     */

    /*
     * Permit references to unknown configuration variable names. We rely on
     * a separate configuration checking tool to spot misspelled names and
     * other kinds of trouble. Enter the configuration directory into the
     * default dictionary.
     */
    if (var_config_dir)
	myfree(var_config_dir);
    if ((config_dir = getenv(CONF_ENV_PATH)) == 0)
	config_dir = DEF_CONFIG_DIR;
    var_config_dir = mystrdup(config_dir);
    set_mail_conf_str(VAR_CONFIG_DIR, var_config_dir);

    /*
     * If the configuration directory name comes from an untrusted source,
     * require that it is listed in the default main.cf file.
     */
    if (strcmp(var_config_dir, DEF_CONFIG_DIR) != 0	/* non-default */
	&& unsafe())				/* untrusted env and cli */
	mail_conf_checkdir(var_config_dir);
    path = concatenate(var_config_dir, "/", "main.cf", (char *) 0);
    if (dict_load_file_xt(CONFIG_DICT, path) == 0)
	msg_fatal("open %s: %m", path);
    myfree(path);
}

/* mail_conf_flush - discard configuration dictionary */

void    mail_conf_flush(void)
{
    if (dict_handle(CONFIG_DICT) != 0)
	dict_unregister(CONFIG_DICT);
}

/* mail_conf_eval - expand macros in string */

const char *mail_conf_eval(const char *string)
{
#define RECURSIVE	1

    return (dict_eval(CONFIG_DICT, string, RECURSIVE));
}

/* mail_conf_eval_once - expand one level of macros in string */

const char *mail_conf_eval_once(const char *string)
{
#define NONRECURSIVE	0

    return (dict_eval(CONFIG_DICT, string, NONRECURSIVE));
}

/* mail_conf_lookup - lookup named variable */

const char *mail_conf_lookup(const char *name)
{
    return (dict_lookup(CONFIG_DICT, name));
}

/* mail_conf_lookup_eval - expand named variable */

const char *mail_conf_lookup_eval(const char *name)
{
    const char *value;

#define RECURSIVE	1

    if ((value = dict_lookup(CONFIG_DICT, name)) != 0)
	value = dict_eval(CONFIG_DICT, value, RECURSIVE);
    return (value);
}

/* mail_conf_update - update parameter */

void    mail_conf_update(const char *key, const char *value)
{
    dict_update(CONFIG_DICT, key, value);
}
