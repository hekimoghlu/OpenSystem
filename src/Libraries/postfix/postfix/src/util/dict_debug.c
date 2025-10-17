/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
/* System libraries. */

#include <sys_defs.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <dict.h>

/* Application-specific. */

typedef struct {
    DICT    dict;			/* the proxy service */
    DICT   *real_dict;			/* encapsulated object */
} DICT_DEBUG;

/* dict_debug_lookup - log lookup operation */

static const char *dict_debug_lookup(DICT *dict, const char *key)
{
    DICT_DEBUG *dict_debug = (DICT_DEBUG *) dict;
    DICT   *real_dict = dict_debug->real_dict;
    const char *result;

    real_dict->flags = dict->flags;
    result = dict_get(real_dict, key);
    dict->flags = real_dict->flags;
    msg_info("%s:%s lookup: \"%s\" = \"%s\"", dict->type, dict->name, key,
	     result ? result : real_dict->error ? "error" : "not_found");
    DICT_ERR_VAL_RETURN(dict, real_dict->error, result);
}

/* dict_debug_update - log update operation */

static int dict_debug_update(DICT *dict, const char *key, const char *value)
{
    DICT_DEBUG *dict_debug = (DICT_DEBUG *) dict;
    DICT   *real_dict = dict_debug->real_dict;
    int     result;

    real_dict->flags = dict->flags;
    result = dict_put(real_dict, key, value);
    dict->flags = real_dict->flags;
    msg_info("%s:%s update: \"%s\" = \"%s\": %s", dict->type, dict->name,
	     key, value, result == 0 ? "success" : real_dict->error ?
	     "error" : "failed");
    DICT_ERR_VAL_RETURN(dict, real_dict->error, result);
}

/* dict_debug_delete - log delete operation */

static int dict_debug_delete(DICT *dict, const char *key)
{
    DICT_DEBUG *dict_debug = (DICT_DEBUG *) dict;
    DICT   *real_dict = dict_debug->real_dict;
    int     result;

    real_dict->flags = dict->flags;
    result = dict_del(real_dict, key);
    dict->flags = real_dict->flags;
    msg_info("%s:%s delete: \"%s\": %s", dict->type, dict->name, key,
	     result == 0 ? "success" : real_dict->error ?
	     "error" : "failed");
    DICT_ERR_VAL_RETURN(dict, real_dict->error, result);
}

/* dict_debug_sequence - log sequence operation */

static int dict_debug_sequence(DICT *dict, int function,
			               const char **key, const char **value)
{
    DICT_DEBUG *dict_debug = (DICT_DEBUG *) dict;
    DICT   *real_dict = dict_debug->real_dict;
    int     result;

    real_dict->flags = dict->flags;
    result = dict_seq(real_dict, function, key, value);
    dict->flags = real_dict->flags;
    if (result == 0)
	msg_info("%s:%s sequence: \"%s\" = \"%s\"", dict->type, dict->name,
		 *key, *value);
    else
	msg_info("%s:%s sequence: found EOF", dict->type, dict->name);
    DICT_ERR_VAL_RETURN(dict, real_dict->error, result);
}

/* dict_debug_close - log operation */

static void dict_debug_close(DICT *dict)
{
    DICT_DEBUG *dict_debug = (DICT_DEBUG *) dict;

    dict_close(dict_debug->real_dict);
    dict_free(dict);
}

/* dict_debug - encapsulate dictionary object and install proxies */

DICT   *dict_debug(DICT *real_dict)
{
    DICT_DEBUG *dict_debug;

    dict_debug = (DICT_DEBUG *) dict_alloc(real_dict->type,
				      real_dict->name, sizeof(*dict_debug));
    dict_debug->dict.flags = real_dict->flags;	/* XXX not synchronized */
    dict_debug->dict.lookup = dict_debug_lookup;
    dict_debug->dict.update = dict_debug_update;
    dict_debug->dict.delete = dict_debug_delete;
    dict_debug->dict.sequence = dict_debug_sequence;
    dict_debug->dict.close = dict_debug_close;
    dict_debug->real_dict = real_dict;
    return (&dict_debug->dict);
}
