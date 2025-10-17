/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
#include <string.h>
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <htable.h>
#include <vstring.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <stringops.h>

/* Global library. */

#include <record.h>
#include <rec_type.h>
#include <is_header.h>
#include <quote_822_local.h>
#include <header_opts.h>
#include <delivered_hdr.h>
#include <fold_addr.h>

 /*
  * Application-specific.
  */
struct DELIVERED_HDR_INFO {
    int     flags;
    VSTRING *buf;
    VSTRING *fold;
    HTABLE *table;
};

#define STR(x) vstring_str(x)

/* delivered_hdr_init - extract delivered-to information from the message */

DELIVERED_HDR_INFO *delivered_hdr_init(VSTREAM *fp, off_t offset, int flags)
{
    char   *cp;
    DELIVERED_HDR_INFO *info;
    const HEADER_OPTS *hdr;

    /*
     * Sanity check.
     */
    info = (DELIVERED_HDR_INFO *) mymalloc(sizeof(*info));
    info->flags = flags;
    info->buf = vstring_alloc(10);
    info->fold = vstring_alloc(10);
    info->table = htable_create(0);

    if (vstream_fseek(fp, offset, SEEK_SET) < 0)
	msg_fatal("seek queue file %s: %m", VSTREAM_PATH(fp));

    /*
     * XXX Assume that mail_copy() produces delivered-to headers that fit in
     * a REC_TYPE_NORM record. Lowercase the delivered-to addresses for
     * consistency.
     * 
     * XXX Don't get bogged down by gazillions of delivered-to headers.
     */
#define DELIVERED_HDR_LIMIT	1000

    while (rec_get(fp, info->buf, 0) == REC_TYPE_NORM
	   && info->table->used < DELIVERED_HDR_LIMIT) {
	if (is_header(STR(info->buf))) {
	    if ((hdr = header_opts_find(STR(info->buf))) != 0
		&& hdr->type == HDR_DELIVERED_TO) {
		cp = STR(info->buf) + strlen(hdr->name) + 1;
		while (ISSPACE(*cp))
		    cp++;
		cp = fold_addr(info->fold, cp, info->flags);
		if (msg_verbose)
		    msg_info("delivered_hdr_init: %s", cp);
		htable_enter(info->table, cp, (void *) 0);
	    }
	} else if (ISSPACE(STR(info->buf)[0])) {
	    continue;
	} else {
	    break;
	}
    }
    return (info);
}

/* delivered_hdr_find - look up recipient in delivered table */

int     delivered_hdr_find(DELIVERED_HDR_INFO *info, const char *address)
{
    HTABLE_INFO *ht;
    const char *addr_key;

    /*
     * mail_copy() uses quote_822_local() when writing the Delivered-To:
     * header. We must therefore apply the same transformation when looking
     * up the recipient. Lowercase the delivered-to address for consistency.
     */
    quote_822_local(info->buf, address);
    addr_key = fold_addr(info->fold, STR(info->buf), info->flags);
    ht = htable_locate(info->table, addr_key);
    return (ht != 0);
}

/* delivered_hdr_free - destructor */

void    delivered_hdr_free(DELIVERED_HDR_INFO *info)
{
    vstring_free(info->buf);
    vstring_free(info->fold);
    htable_free(info->table, (void (*) (void *)) 0);
    myfree((void *) info);
}
