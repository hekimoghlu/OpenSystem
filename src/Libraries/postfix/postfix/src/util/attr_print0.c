/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstream.h>
#include <htable.h>
#include <attr.h>
#include <base64_code.h>

#define STR(x) vstring_str(x)
#define LEN(x) VSTRING_LEN(x)

/* attr_vprint0 - send attribute list to stream */

int     attr_vprint0(VSTREAM *fp, int flags, va_list ap)
{
    const char *myname = "attr_print0";
    int     attr_type;
    char   *attr_name;
    unsigned int_val;
    unsigned long long_val;
    char   *str_val;
    HTABLE_INFO **ht_info_list;
    HTABLE_INFO **ht;
    ssize_t len_val;
    static VSTRING *base64_buf;
    ATTR_PRINT_SLAVE_FN print_fn;
    void   *print_arg;

    /*
     * Sanity check.
     */
    if (flags & ~ATTR_FLAG_ALL)
	msg_panic("%s: bad flags: 0x%x", myname, flags);

    /*
     * Iterate over all (type, name, value) triples, and produce output on
     * the fly.
     */
    while ((attr_type = va_arg(ap, int)) != ATTR_TYPE_END) {
	switch (attr_type) {
	case ATTR_TYPE_INT:
	    attr_name = va_arg(ap, char *);
	    vstream_fwrite(fp, attr_name, strlen(attr_name) + 1);
	    int_val = va_arg(ap, int);
	    vstream_fprintf(fp, "%u", (unsigned) int_val);
	    VSTREAM_PUTC('\0', fp);
	    if (msg_verbose)
		msg_info("send attr %s = %u", attr_name, int_val);
	    break;
	case ATTR_TYPE_LONG:
	    attr_name = va_arg(ap, char *);
	    vstream_fwrite(fp, attr_name, strlen(attr_name) + 1);
	    long_val = va_arg(ap, unsigned long);
	    vstream_fprintf(fp, "%lu", (unsigned long) long_val);
	    VSTREAM_PUTC('\0', fp);
	    if (msg_verbose)
		msg_info("send attr %s = %lu", attr_name, long_val);
	    break;
	case ATTR_TYPE_STR:
	    attr_name = va_arg(ap, char *);
	    vstream_fwrite(fp, attr_name, strlen(attr_name) + 1);
	    str_val = va_arg(ap, char *);
	    vstream_fwrite(fp, str_val, strlen(str_val) + 1);
	    if (msg_verbose)
		msg_info("send attr %s = %s", attr_name, str_val);
	    break;
	case ATTR_TYPE_DATA:
	    attr_name = va_arg(ap, char *);
	    vstream_fwrite(fp, attr_name, strlen(attr_name) + 1);
	    len_val = va_arg(ap, ssize_t);
	    str_val = va_arg(ap, char *);
	    if (base64_buf == 0)
		base64_buf = vstring_alloc(10);
	    base64_encode(base64_buf, str_val, len_val);
	    vstream_fwrite(fp, STR(base64_buf), LEN(base64_buf) + 1);
	    if (msg_verbose)
		msg_info("send attr %s = [data %ld bytes]",
			 attr_name, (long) len_val);
	    break;
	case ATTR_TYPE_FUNC:
	    print_fn = va_arg(ap, ATTR_PRINT_SLAVE_FN);
	    print_arg = va_arg(ap, void *);
	    print_fn(attr_print0, fp, flags | ATTR_FLAG_MORE, print_arg);
	    break;
	case ATTR_TYPE_HASH:
	    vstream_fwrite(fp, ATTR_NAME_OPEN, sizeof(ATTR_NAME_OPEN));
	    ht_info_list = htable_list(va_arg(ap, HTABLE *));
	    for (ht = ht_info_list; *ht; ht++) {
		vstream_fwrite(fp, ht[0]->key, strlen(ht[0]->key) + 1);
		vstream_fwrite(fp, ht[0]->value, strlen(ht[0]->value) + 1);
		if (msg_verbose)
		    msg_info("send attr name %s value %s",
			     ht[0]->key, (char *) ht[0]->value);
	    }
	    myfree((void *) ht_info_list);
	    vstream_fwrite(fp, ATTR_NAME_CLOSE, sizeof(ATTR_NAME_CLOSE));
	    break;
	default:
	    msg_panic("%s: unknown type code: %d", myname, attr_type);
	}
    }
    if ((flags & ATTR_FLAG_MORE) == 0)
	VSTREAM_PUTC('\0', fp);
    return (vstream_ferror(fp));
}

int     attr_print0(VSTREAM *fp, int flags,...)
{
    va_list ap;
    int     ret;

    va_start(ap, flags);
    ret = attr_vprint0(fp, flags, ap);
    va_end(ap);
    return (ret);
}

#ifdef TEST

 /*
  * Proof of concept test program.  Mirror image of the attr_scan0 test
  * program.
  */
#include <msg_vstream.h>

int     main(int unused_argc, char **argv)
{
    HTABLE *table = htable_create(1);

    msg_vstream_init(argv[0], VSTREAM_ERR);
    msg_verbose = 1;
    htable_enter(table, "foo-name", mystrdup("foo-value"));
    htable_enter(table, "bar-name", mystrdup("bar-value"));
    attr_print0(VSTREAM_OUT, ATTR_FLAG_NONE,
		SEND_ATTR_INT(ATTR_NAME_INT, 4711),
		SEND_ATTR_LONG(ATTR_NAME_LONG, 1234L),
		SEND_ATTR_STR(ATTR_NAME_STR, "whoopee"),
		SEND_ATTR_DATA(ATTR_NAME_DATA, strlen("whoopee"), "whoopee"),
		SEND_ATTR_HASH(table),
		SEND_ATTR_LONG(ATTR_NAME_LONG, 4321L),
		ATTR_TYPE_END);
    attr_print0(VSTREAM_OUT, ATTR_FLAG_NONE,
		SEND_ATTR_INT(ATTR_NAME_INT, 4711),
		SEND_ATTR_LONG(ATTR_NAME_LONG, 1234L),
		SEND_ATTR_STR(ATTR_NAME_STR, "whoopee"),
		SEND_ATTR_DATA(ATTR_NAME_DATA, strlen("whoopee"), "whoopee"),
		ATTR_TYPE_END);
    if (vstream_fflush(VSTREAM_OUT) != 0)
	msg_fatal("write error: %m");

    htable_free(table, myfree);
    return (0);
}

#endif
