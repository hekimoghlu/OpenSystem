/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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

#include "sys_defs.h"

#include <stdlib.h>
#include <errno.h>
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include "msg.h"
#include "mymalloc.h"
#include "vstring.h"
#include "dict.h"

/* Global library. */

#include "mail_conf.h"

/* Application-specific. */

#include "cfg_parser.h"

/* get string from file */

static char *get_dict_str(const struct CFG_PARSER *parser,
			          const char *name, const char *defval,
			          int min, int max)
{
    const char *strval;
    int     len;

    if ((strval = dict_lookup(parser->name, name)) == 0)
	strval = defval;

    len = strlen(strval);
    if (min && len < min)
	msg_fatal("%s: bad string length %d < %d: %s = %s",
		  parser->name, len, min, name, strval);
    if (max && len > max)
	msg_fatal("%s: bad string length %d > %d: %s = %s",
		  parser->name, len, max, name, strval);
    return (mystrdup(strval));
}

/* get string from main.cf */

static char *get_main_str(const struct CFG_PARSER *parser,
			          const char *name, const char *defval,
			          int min, int max)
{
    static VSTRING *buf = 0;

    if (buf == 0)
	buf = vstring_alloc(15);
    vstring_sprintf(buf, "%s_%s", parser->name, name);
    return (get_mail_conf_str(vstring_str(buf), defval, min, max));
}

/* get integer from file */

static int get_dict_int(const struct CFG_PARSER *parser,
		             const char *name, int defval, int min, int max)
{
    const char *strval;
    char   *end;
    int     intval;
    long    longval;

    if ((strval = dict_lookup(parser->name, name)) != 0) {
	errno = 0;
	intval = longval = strtol(strval, &end, 10);
	if (*strval == 0 || *end != 0 || errno == ERANGE || longval != intval)
	    msg_fatal("%s: bad numerical configuration: %s = %s",
		      parser->name, name, strval);
    } else
	intval = defval;
    if (min && intval < min)
	msg_fatal("%s: invalid %s parameter value %d < %d",
		  parser->name, name, intval, min);
    if (max && intval > max)
	msg_fatal("%s: invalid %s parameter value %d > %d",
		  parser->name, name, intval, max);
    return (intval);
}

/* get integer from main.cf */

static int get_main_int(const struct CFG_PARSER *parser,
		             const char *name, int defval, int min, int max)
{
    static VSTRING *buf = 0;

    if (buf == 0)
	buf = vstring_alloc(15);
    vstring_sprintf(buf, "%s_%s", parser->name, name);
    return (get_mail_conf_int(vstring_str(buf), defval, min, max));
}

/* get boolean option from file */

static int get_dict_bool(const struct CFG_PARSER *parser,
			         const char *name, int defval)
{
    const char *strval;
    int     intval;

    if ((strval = dict_lookup(parser->name, name)) != 0) {
	if (strcasecmp(strval, CONFIG_BOOL_YES) == 0) {
	    intval = 1;
	} else if (strcasecmp(strval, CONFIG_BOOL_NO) == 0) {
	    intval = 0;
	} else {
	    msg_fatal("%s: bad boolean configuration: %s = %s",
		      parser->name, name, strval);
	}
    } else
	intval = defval;
    return (intval);
}

/* get boolean option from main.cf */

static int get_main_bool(const struct CFG_PARSER *parser,
			         const char *name, int defval)
{
    static VSTRING *buf = 0;

    if (buf == 0)
	buf = vstring_alloc(15);
    vstring_sprintf(buf, "%s_%s", parser->name, name);
    return (get_mail_conf_bool(vstring_str(buf), defval));
}

/* initialize parser */

CFG_PARSER *cfg_parser_alloc(const char *pname)
{
    const char *myname = "cfg_parser_alloc";
    CFG_PARSER *parser;
    DICT   *dict;

    if (pname == 0 || *pname == 0)
	msg_fatal("%s: null parser name", myname);
    parser = (CFG_PARSER *) mymalloc(sizeof(*parser));
    parser->name = mystrdup(pname);
    if (*parser->name == '/' || *parser->name == '.') {
	if (dict_load_file_xt(parser->name, parser->name) == 0) {
	    myfree(parser->name);
	    myfree((void *) parser);
	    return (0);
	}
	parser->get_str = get_dict_str;
	parser->get_int = get_dict_int;
	parser->get_bool = get_dict_bool;
	dict = dict_handle(parser->name);
    } else {
	parser->get_str = get_main_str;
	parser->get_int = get_main_int;
	parser->get_bool = get_main_bool;
	dict = dict_handle(CONFIG_DICT);	/* XXX Use proper API */
    }
    if (dict == 0)
	msg_panic("%s: dict_handle failed", myname);
    parser->owner = dict->owner;
    return (parser);
}

/* get string */

char   *cfg_get_str(const CFG_PARSER *parser, const char *name,
		            const char *defval, int min, int max)
{
    const char *myname = "cfg_get_str";
    char   *strval;

    strval = parser->get_str(parser, name, (defval ? defval : ""), min, max);
    if (defval == 0 && *strval == 0) {
	/* the caller wants NULL instead of "" */
	myfree(strval);
	strval = 0;
    }
    if (msg_verbose)
	msg_info("%s: %s: %s = %s", myname, parser->name, name,
		 (strval ? strval : "<NULL>"));
    return (strval);
}

/* get integer */

int     cfg_get_int(const CFG_PARSER *parser, const char *name, int defval,
		            int min, int max)
{
    const char *myname = "cfg_get_int";
    int     intval;

    intval = parser->get_int(parser, name, defval, min, max);
    if (msg_verbose)
	msg_info("%s: %s: %s = %d", myname, parser->name, name, intval);
    return (intval);
}

/* get boolean option */

int     cfg_get_bool(const CFG_PARSER *parser, const char *name, int defval)
{
    const char *myname = "cfg_get_bool";
    int     intval;

    intval = parser->get_bool(parser, name, defval);
    if (msg_verbose)
	msg_info("%s: %s: %s = %s", myname, parser->name, name,
		 (intval ? "on" : "off"));
    return (intval);
}

/* release parser */

CFG_PARSER *cfg_parser_free(CFG_PARSER *parser)
{
    const char *myname = "cfg_parser_free";

    if (parser->name == 0 || *parser->name == 0)
	msg_panic("%s: null parser name", myname);
    if (*parser->name == '/' || *parser->name == '.') {
	if (dict_handle(parser->name))
	    dict_unregister(parser->name);
    }
    myfree(parser->name);
    myfree((void *) parser);
    return (0);
}
