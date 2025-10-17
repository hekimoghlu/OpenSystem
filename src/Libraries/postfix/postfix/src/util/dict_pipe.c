/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
#include "mymalloc.h"
#include "htable.h"
#include "dict.h"
#include "dict_pipe.h"
#include "stringops.h"
#include "vstring.h"

/* Application-specific. */

typedef struct {
    DICT    dict;			/* generic members */
    ARGV   *map_pipe;			/* pipelined tables */
    VSTRING *qr_buf;			/* query/reply buffer */
} DICT_PIPE;

#define STR(x) vstring_str(x)

/* dict_pipe_lookup - search pipelined tables */

static const char *dict_pipe_lookup(DICT *dict, const char *query)
{
    static const char myname[] = "dict_pipe_lookup";
    DICT_PIPE *dict_pipe = (DICT_PIPE *) dict;
    DICT   *map;
    char  **cpp;
    char   *dict_type_name;
    const char *result = 0;

    vstring_strcpy(dict_pipe->qr_buf, query);
    for (cpp = dict_pipe->map_pipe->argv; (dict_type_name = *cpp) != 0; cpp++) {
	if ((map = dict_handle(dict_type_name)) == 0)
	    msg_panic("%s: dictionary \"%s\" not found", myname, dict_type_name);
	if ((result = dict_get(map, STR(dict_pipe->qr_buf))) == 0)
	    DICT_ERR_VAL_RETURN(dict, map->error, result);
	vstring_strcpy(dict_pipe->qr_buf, result);
    }
    DICT_ERR_VAL_RETURN(dict, DICT_ERR_NONE, STR(dict_pipe->qr_buf));
}

/* dict_pipe_close - disassociate from pipelined tables */

static void dict_pipe_close(DICT *dict)
{
    DICT_PIPE *dict_pipe = (DICT_PIPE *) dict;
    char  **cpp;
    char   *dict_type_name;

    for (cpp = dict_pipe->map_pipe->argv; (dict_type_name = *cpp) != 0; cpp++)
	dict_unregister(dict_type_name);
    argv_free(dict_pipe->map_pipe);
    vstring_free(dict_pipe->qr_buf);
    dict_free(dict);
}

/* dict_pipe_open - open pipelined tables */

DICT   *dict_pipe_open(const char *name, int open_flags, int dict_flags)
{
    static const char myname[] = "dict_pipe_open";
    DICT_PIPE *dict_pipe;
    char   *saved_name = 0;
    char   *dict_type_name;
    ARGV   *argv = 0;
    char  **cpp;
    DICT   *dict;
    int     match_flags = 0;
    struct DICT_OWNER aggr_owner;
    size_t  len;

    /*
     * Clarity first. Let the optimizer worry about redundant code.
     */
#define DICT_PIPE_RETURN(x) do { \
	    if (saved_name != 0) \
		myfree(saved_name); \
	    if (argv != 0) \
		argv_free(argv); \
	    return (x); \
	} while (0)

    /*
     * Sanity checks.
     */
    if (open_flags != O_RDONLY)
	DICT_PIPE_RETURN(dict_surrogate(DICT_TYPE_PIPE, name,
					open_flags, dict_flags,
				  "%s:%s map requires O_RDONLY access mode",
					DICT_TYPE_PIPE, name));

    /*
     * Split the table name into its constituent parts.
     */
    if ((len = balpar(name, CHARS_BRACE)) == 0 || name[len] != 0
	|| *(saved_name = mystrndup(name + 1, len - 2)) == 0
	|| ((argv = argv_splitq(saved_name, CHARS_COMMA_SP, CHARS_BRACE)),
	    (argv->argc == 0)))
	DICT_PIPE_RETURN(dict_surrogate(DICT_TYPE_PIPE, name,
					open_flags, dict_flags,
					"bad syntax: \"%s:%s\"; "
					"need \"%s:{type:name...}\"",
					DICT_TYPE_PIPE, name,
					DICT_TYPE_PIPE));

    /*
     * The least-trusted table in the pipeline determines the over-all trust
     * level. The first table determines the pattern-matching flags.
     */
    DICT_OWNER_AGGREGATE_INIT(aggr_owner);
    for (cpp = argv->argv; (dict_type_name = *cpp) != 0; cpp++) {
	if (msg_verbose)
	    msg_info("%s: %s", myname, dict_type_name);
	if (strchr(dict_type_name, ':') == 0)
	    DICT_PIPE_RETURN(dict_surrogate(DICT_TYPE_PIPE, name,
					    open_flags, dict_flags,
					    "bad syntax: \"%s:%s\"; "
					    "need \"%s:{type:name...}\"",
					    DICT_TYPE_PIPE, name,
					    DICT_TYPE_PIPE));
	if ((dict = dict_handle(dict_type_name)) == 0)
	    dict = dict_open(dict_type_name, open_flags, dict_flags);
	dict_register(dict_type_name, dict);
	DICT_OWNER_AGGREGATE_UPDATE(aggr_owner, dict->owner);
	if (cpp == argv->argv)
	    match_flags = dict->flags & (DICT_FLAG_FIXED | DICT_FLAG_PATTERN);
    }

    /*
     * Bundle up the result.
     */
    dict_pipe =
	(DICT_PIPE *) dict_alloc(DICT_TYPE_PIPE, name, sizeof(*dict_pipe));
    dict_pipe->dict.lookup = dict_pipe_lookup;
    dict_pipe->dict.close = dict_pipe_close;
    dict_pipe->dict.flags = dict_flags | match_flags;
    dict_pipe->dict.owner = aggr_owner;
    dict_pipe->qr_buf = vstring_alloc(100);
    dict_pipe->map_pipe = argv;
    argv = 0;
    DICT_PIPE_RETURN(DICT_DEBUG (&dict_pipe->dict));
}
