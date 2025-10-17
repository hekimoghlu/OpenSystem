/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#include <stdarg.h>
#include <string.h>
#include <stdio.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstream.h>
#include <vstring.h>
#include <htable.h>
#include <base64_code.h>
#include <attr.h>

/* Application specific. */

#define STR(x)	vstring_str(x)
#define LEN(x)	VSTRING_LEN(x)

/* attr_scan64_string - pull a string from the input stream */

static int attr_scan64_string(VSTREAM *fp, VSTRING *plain_buf, const char *context)
{
    static VSTRING *base64_buf = 0;

#if 0
    extern int var_line_limit;		/* XXX */
    int     limit = var_line_limit * 4;

#endif
    int     ch;

    if (base64_buf == 0)
	base64_buf = vstring_alloc(10);

    VSTRING_RESET(base64_buf);
    while ((ch = VSTREAM_GETC(fp)) != ':' && ch != '\n') {
	if (ch == VSTREAM_EOF) {
	    msg_warn("%s on %s while reading %s",
		vstream_ftimeout(fp) ? "timeout" : "premature end-of-input",
		     VSTREAM_PATH(fp), context);
	    return (-1);
	}
	VSTRING_ADDCH(base64_buf, ch);
#if 0
	if (LEN(base64_buf) > limit) {
	    msg_warn("string length > %d characters from %s while reading %s",
		     limit, VSTREAM_PATH(fp), context);
	    return (-1);
	}
#endif
    }
    VSTRING_TERMINATE(base64_buf);
    if (base64_decode(plain_buf, STR(base64_buf), LEN(base64_buf)) == 0) {
	msg_warn("malformed base64 data from %s: %.100s",
		 VSTREAM_PATH(fp), STR(base64_buf));
	return (-1);
    }
    if (msg_verbose)
	msg_info("%s: %s", context, *STR(plain_buf) ? STR(plain_buf) : "(end)");
    return (ch);
}

/* attr_scan64_number - pull a number from the input stream */

static int attr_scan64_number(VSTREAM *fp, unsigned *ptr, VSTRING *str_buf,
			              const char *context)
{
    char    junk = 0;
    int     ch;

    if ((ch = attr_scan64_string(fp, str_buf, context)) < 0)
	return (-1);
    if (sscanf(STR(str_buf), "%u%c", ptr, &junk) != 1 || junk != 0) {
	msg_warn("malformed numerical data from %s while reading %s: %.100s",
		 VSTREAM_PATH(fp), context, STR(str_buf));
	return (-1);
    }
    return (ch);
}

/* attr_scan64_long_number - pull a number from the input stream */

static int attr_scan64_long_number(VSTREAM *fp, unsigned long *ptr,
				           VSTRING *str_buf,
				           const char *context)
{
    char    junk = 0;
    int     ch;

    if ((ch = attr_scan64_string(fp, str_buf, context)) < 0)
	return (-1);
    if (sscanf(STR(str_buf), "%lu%c", ptr, &junk) != 1 || junk != 0) {
	msg_warn("malformed numerical data from %s while reading %s: %.100s",
		 VSTREAM_PATH(fp), context, STR(str_buf));
	return (-1);
    }
    return (ch);
}

/* attr_vscan64 - receive attribute list from stream */

int     attr_vscan64(VSTREAM *fp, int flags, va_list ap)
{
    const char *myname = "attr_scan64";
    static VSTRING *str_buf = 0;
    static VSTRING *name_buf = 0;
    int     wanted_type = -1;
    char   *wanted_name;
    unsigned int *number;
    unsigned long *long_number;
    VSTRING *string;
    HTABLE *hash_table;
    int     ch;
    int     conversions;
    ATTR_SCAN_SLAVE_FN scan_fn;
    void   *scan_arg;

    /*
     * Sanity check.
     */
    if (flags & ~ATTR_FLAG_ALL)
	msg_panic("%s: bad flags: 0x%x", myname, flags);

    /*
     * EOF check.
     */
    if ((ch = VSTREAM_GETC(fp)) == VSTREAM_EOF)
	return (0);
    vstream_ungetc(fp, ch);

    /*
     * Initialize.
     */
    if (str_buf == 0) {
	str_buf = vstring_alloc(10);
	name_buf = vstring_alloc(10);
    }

    /*
     * Iterate over all (type, name, value) triples.
     */
    for (conversions = 0; /* void */ ; conversions++) {

	/*
	 * Determine the next attribute type and attribute name on the
	 * caller's wish list.
	 * 
	 * If we're reading into a hash table, we already know that the
	 * attribute value is string-valued, and we get the attribute name
	 * from the input stream instead. This is secure only when the
	 * resulting table is queried with known to be good attribute names.
	 */
	if (wanted_type != ATTR_TYPE_HASH
	    && wanted_type != ATTR_TYPE_CLOSE) {
	    wanted_type = va_arg(ap, int);
	    if (wanted_type == ATTR_TYPE_END) {
		if ((flags & ATTR_FLAG_MORE) != 0)
		    return (conversions);
		wanted_name = "(list terminator)";
	    } else if (wanted_type == ATTR_TYPE_HASH) {
		wanted_name = "(any attribute name or list terminator)";
		hash_table = va_arg(ap, HTABLE *);
	    } else if (wanted_type != ATTR_TYPE_FUNC) {
		wanted_name = va_arg(ap, char *);
	    }
	}

	/*
	 * Locate the next attribute of interest in the input stream.
	 */
	while (wanted_type != ATTR_TYPE_FUNC) {

	    /*
	     * Get the name of the next attribute. Hitting EOF is always bad.
	     * Hitting the end-of-input early is OK if the caller is prepared
	     * to deal with missing inputs.
	     */
	    if (msg_verbose)
		msg_info("%s: wanted attribute: %s",
			 VSTREAM_PATH(fp), wanted_name);
	    if ((ch = attr_scan64_string(fp, name_buf,
				    "input attribute name")) == VSTREAM_EOF)
		return (-1);
	    if (ch == '\n' && LEN(name_buf) == 0) {
		if (wanted_type == ATTR_TYPE_END
		    || wanted_type == ATTR_TYPE_HASH)
		    return (conversions);
		if ((flags & ATTR_FLAG_MISSING) != 0)
		    msg_warn("missing attribute %s in input from %s",
			     wanted_name, VSTREAM_PATH(fp));
		return (conversions);
	    }

	    /*
	     * See if the caller asks for this attribute.
	     */
	    if (wanted_type == ATTR_TYPE_HASH
	      && ch == '\n' && strcmp(ATTR_NAME_OPEN, STR(name_buf)) == 0) {
		wanted_type = ATTR_TYPE_CLOSE;
		wanted_name = "(any attribute name or '}')";
		/* Advance in the input stream. */
		continue;
	    } else if (wanted_type == ATTR_TYPE_CLOSE
	     && ch == '\n' && strcmp(ATTR_NAME_CLOSE, STR(name_buf)) == 0) {
		/* Advance in the argument list. */
		wanted_type = -1;
		break;
	    }
	    if (wanted_type == ATTR_TYPE_HASH
		|| wanted_type == ATTR_TYPE_CLOSE
		|| (wanted_type != ATTR_TYPE_END
		    && strcmp(wanted_name, STR(name_buf)) == 0))
		break;
	    if ((flags & ATTR_FLAG_EXTRA) != 0) {
		msg_warn("unexpected attribute %s from %s (expecting: %s)",
			 STR(name_buf), VSTREAM_PATH(fp), wanted_name);
		return (conversions);
	    }

	    /*
	     * Skip over this attribute. The caller does not ask for it.
	     */
	    while (ch != '\n' && (ch = VSTREAM_GETC(fp)) != VSTREAM_EOF)
		 /* void */ ;
	}

	/*
	 * Do the requested conversion. If the target attribute is a
	 * non-array type, disallow sending a multi-valued attribute, and
	 * disallow sending no value. If the target attribute is an array
	 * type, allow the sender to send a zero-element array (i.e. no value
	 * at all). XXX Need to impose a bound on the number of array
	 * elements.
	 */
	switch (wanted_type) {
	case ATTR_TYPE_INT:
	    if (ch != ':') {
		msg_warn("missing value for number attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    number = va_arg(ap, unsigned int *);
	    if ((ch = attr_scan64_number(fp, number, str_buf,
					 "input attribute value")) < 0)
		return (-1);
	    if (ch != '\n') {
		msg_warn("multiple values for attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    break;
	case ATTR_TYPE_LONG:
	    if (ch != ':') {
		msg_warn("missing value for number attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    long_number = va_arg(ap, unsigned long *);
	    if ((ch = attr_scan64_long_number(fp, long_number, str_buf,
					      "input attribute value")) < 0)
		return (-1);
	    if (ch != '\n') {
		msg_warn("multiple values for attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    break;
	case ATTR_TYPE_STR:
	    if (ch != ':') {
		msg_warn("missing value for string attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    string = va_arg(ap, VSTRING *);
	    if ((ch = attr_scan64_string(fp, string,
					 "input attribute value")) < 0)
		return (-1);
	    if (ch != '\n') {
		msg_warn("multiple values for attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    break;
	case ATTR_TYPE_DATA:
	    if (ch != ':') {
		msg_warn("missing value for data attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    string = va_arg(ap, VSTRING *);
	    if ((ch = attr_scan64_string(fp, string,
					 "input attribute value")) < 0)
		return (-1);
	    if (ch != '\n') {
		msg_warn("multiple values for attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    break;
	case ATTR_TYPE_FUNC:
	    scan_fn = va_arg(ap, ATTR_SCAN_SLAVE_FN);
	    scan_arg = va_arg(ap, void *);
	    if (scan_fn(attr_scan64, fp, flags | ATTR_FLAG_MORE, scan_arg) < 0)
		return (-1);
	    break;
	case ATTR_TYPE_HASH:
	case ATTR_TYPE_CLOSE:
	    if (ch != ':') {
		msg_warn("missing value for string attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    if ((ch = attr_scan64_string(fp, str_buf,
					 "input attribute value")) < 0)
		return (-1);
	    if (ch != '\n') {
		msg_warn("multiple values for attribute %s from %s",
			 STR(name_buf), VSTREAM_PATH(fp));
		return (-1);
	    }
	    if (htable_locate(hash_table, STR(name_buf)) != 0) {
		if ((flags & ATTR_FLAG_EXTRA) != 0) {
		    msg_warn("duplicate attribute %s in input from %s",
			     STR(name_buf), VSTREAM_PATH(fp));
		    return (conversions);
		}
	    } else if (hash_table->used >= ATTR_HASH_LIMIT) {
		msg_warn("attribute count exceeds limit %d in input from %s",
			 ATTR_HASH_LIMIT, VSTREAM_PATH(fp));
		return (conversions);
	    } else {
		htable_enter(hash_table, STR(name_buf),
			     mystrdup(STR(str_buf)));
	    }
	    break;
	case -1:
	    conversions -= 1;
	    break;
	default:
	    msg_panic("%s: unknown type code: %d", myname, wanted_type);
	}
    }
}

/* attr_scan64 - read attribute list from stream */

int     attr_scan64(VSTREAM *fp, int flags,...)
{
    va_list ap;
    int     ret;

    va_start(ap, flags);
    ret = attr_vscan64(fp, flags, ap);
    va_end(ap);
    return (ret);
}

/* attr_scan_more64 - look ahead for more */

int     attr_scan_more64(VSTREAM *fp)
{
    int     ch;

    switch (ch = VSTREAM_GETC(fp)) {
    case '\n':
	if (msg_verbose)
	    msg_info("%s: terminator (consumed)", VSTREAM_PATH(fp));
	return (0);
    case VSTREAM_EOF:
	if (msg_verbose)
	    msg_info("%s: EOF", VSTREAM_PATH(fp));
	return (-1);
    default:
	if (msg_verbose)
	    msg_info("%s: non-terminator '%c' (lookahead)",
		     VSTREAM_PATH(fp), ch);
	(void) vstream_ungetc(fp, ch);
	return (1);
    }
}

#ifdef TEST

 /*
  * Proof of concept test program.  Mirror image of the attr_scan64 test
  * program.
  */
#include <msg_vstream.h>

int     var_line_limit = 2048;

int     main(int unused_argc, char **used_argv)
{
    VSTRING *data_val = vstring_alloc(1);
    VSTRING *str_val = vstring_alloc(1);
    HTABLE *table = htable_create(1);
    HTABLE_INFO **ht_info_list;
    HTABLE_INFO **ht;
    int     int_val;
    long    long_val;
    long    long_val2;
    int     ret;

    msg_verbose = 1;
    msg_vstream_init(used_argv[0], VSTREAM_ERR);
    if ((ret = attr_scan64(VSTREAM_IN,
			   ATTR_FLAG_STRICT,
			   RECV_ATTR_INT(ATTR_NAME_INT, &int_val),
			   RECV_ATTR_LONG(ATTR_NAME_LONG, &long_val),
			   RECV_ATTR_STR(ATTR_NAME_STR, str_val),
			   RECV_ATTR_DATA(ATTR_NAME_DATA, data_val),
			   RECV_ATTR_HASH(table),
			   RECV_ATTR_LONG(ATTR_NAME_LONG, &long_val2),
			   ATTR_TYPE_END)) > 4) {
	vstream_printf("%s %d\n", ATTR_NAME_INT, int_val);
	vstream_printf("%s %ld\n", ATTR_NAME_LONG, long_val);
	vstream_printf("%s %s\n", ATTR_NAME_STR, STR(str_val));
	vstream_printf("%s %s\n", ATTR_NAME_DATA, STR(data_val));
	ht_info_list = htable_list(table);
	for (ht = ht_info_list; *ht; ht++)
	    vstream_printf("(hash) %s %s\n", ht[0]->key, (char *) ht[0]->value);
	myfree((void *) ht_info_list);
	vstream_printf("%s %ld\n", ATTR_NAME_LONG, long_val2);
    } else {
	vstream_printf("return: %d\n", ret);
    }
    if ((ret = attr_scan64(VSTREAM_IN,
			   ATTR_FLAG_STRICT,
			   RECV_ATTR_INT(ATTR_NAME_INT, &int_val),
			   RECV_ATTR_LONG(ATTR_NAME_LONG, &long_val),
			   RECV_ATTR_STR(ATTR_NAME_STR, str_val),
			   RECV_ATTR_DATA(ATTR_NAME_DATA, data_val),
			   ATTR_TYPE_END)) == 4) {
	vstream_printf("%s %d\n", ATTR_NAME_INT, int_val);
	vstream_printf("%s %ld\n", ATTR_NAME_LONG, long_val);
	vstream_printf("%s %s\n", ATTR_NAME_STR, STR(str_val));
	vstream_printf("%s %s\n", ATTR_NAME_DATA, STR(data_val));
	ht_info_list = htable_list(table);
	for (ht = ht_info_list; *ht; ht++)
	    vstream_printf("(hash) %s %s\n", ht[0]->key, (char *) ht[0]->value);
	myfree((void *) ht_info_list);
    } else {
	vstream_printf("return: %d\n", ret);
    }
    if (vstream_fflush(VSTREAM_OUT) != 0)
	msg_fatal("write error: %m");

    vstring_free(data_val);
    vstring_free(str_val);
    htable_free(table, myfree);

    return (0);
}

#endif
