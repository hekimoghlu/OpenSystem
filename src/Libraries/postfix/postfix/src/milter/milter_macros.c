/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#include <attr.h>
#include <mymalloc.h>
#include <vstring.h>

/* Global library. */

#include <mail_proto.h>
#include <milter.h>

 /*
  * Ad-hoc protocol to send/receive milter macro name lists.
  */
#define MAIL_ATTR_MILT_MAC_CONN	"conn_macros"
#define MAIL_ATTR_MILT_MAC_HELO	"helo_macros"
#define MAIL_ATTR_MILT_MAC_MAIL	"mail_macros"
#define MAIL_ATTR_MILT_MAC_RCPT	"rcpt_macros"
#define MAIL_ATTR_MILT_MAC_DATA	"data_macros"
#define MAIL_ATTR_MILT_MAC_EOH	"eoh_macros"
#define MAIL_ATTR_MILT_MAC_EOD	"eod_macros"
#define MAIL_ATTR_MILT_MAC_UNK	"unk_macros"

/* milter_macros_print - write macros structure to stream */

int     milter_macros_print(ATTR_PRINT_MASTER_FN print_fn, VSTREAM *fp,
			            int flags, void *ptr)
{
    MILTER_MACROS *mp = (MILTER_MACROS *) ptr;
    int     ret;

    /*
     * The attribute order does not matter, except that it must be the same
     * as in the milter_macros_scan() function.
     */
    ret = print_fn(fp, flags | ATTR_FLAG_MORE,
		   SEND_ATTR_STR(MAIL_ATTR_MILT_MAC_CONN, mp->conn_macros),
		   SEND_ATTR_STR(MAIL_ATTR_MILT_MAC_HELO, mp->helo_macros),
		   SEND_ATTR_STR(MAIL_ATTR_MILT_MAC_MAIL, mp->mail_macros),
		   SEND_ATTR_STR(MAIL_ATTR_MILT_MAC_RCPT, mp->rcpt_macros),
		   SEND_ATTR_STR(MAIL_ATTR_MILT_MAC_DATA, mp->data_macros),
		   SEND_ATTR_STR(MAIL_ATTR_MILT_MAC_EOH, mp->eoh_macros),
		   SEND_ATTR_STR(MAIL_ATTR_MILT_MAC_EOD, mp->eod_macros),
		   SEND_ATTR_STR(MAIL_ATTR_MILT_MAC_UNK, mp->unk_macros),
		   ATTR_TYPE_END);
    return (ret);
}

/* milter_macros_scan - receive macros structure from stream */

int     milter_macros_scan(ATTR_SCAN_MASTER_FN scan_fn, VSTREAM *fp,
			           int flags, void *ptr)
{
    MILTER_MACROS *mp = (MILTER_MACROS *) ptr;
    int     ret;

    /*
     * We could simplify this by moving memory allocation into attr_scan*().
     */
    VSTRING *conn_macros = vstring_alloc(10);
    VSTRING *helo_macros = vstring_alloc(10);
    VSTRING *mail_macros = vstring_alloc(10);
    VSTRING *rcpt_macros = vstring_alloc(10);
    VSTRING *data_macros = vstring_alloc(10);
    VSTRING *eoh_macros = vstring_alloc(10);
    VSTRING *eod_macros = vstring_alloc(10);
    VSTRING *unk_macros = vstring_alloc(10);

    /*
     * The attribute order does not matter, except that it must be the same
     * as in the milter_macros_print() function.
     */
    ret = scan_fn(fp, flags | ATTR_FLAG_MORE,
		  RECV_ATTR_STR(MAIL_ATTR_MILT_MAC_CONN, conn_macros),
		  RECV_ATTR_STR(MAIL_ATTR_MILT_MAC_HELO, helo_macros),
		  RECV_ATTR_STR(MAIL_ATTR_MILT_MAC_MAIL, mail_macros),
		  RECV_ATTR_STR(MAIL_ATTR_MILT_MAC_RCPT, rcpt_macros),
		  RECV_ATTR_STR(MAIL_ATTR_MILT_MAC_DATA, data_macros),
		  RECV_ATTR_STR(MAIL_ATTR_MILT_MAC_EOH, eoh_macros),
		  RECV_ATTR_STR(MAIL_ATTR_MILT_MAC_EOD, eod_macros),
		  RECV_ATTR_STR(MAIL_ATTR_MILT_MAC_UNK, unk_macros),
		  ATTR_TYPE_END);

    /*
     * Don't optimize for error.
     */
    mp->conn_macros = vstring_export(conn_macros);
    mp->helo_macros = vstring_export(helo_macros);
    mp->mail_macros = vstring_export(mail_macros);
    mp->rcpt_macros = vstring_export(rcpt_macros);
    mp->data_macros = vstring_export(data_macros);
    mp->eoh_macros = vstring_export(eoh_macros);
    mp->eod_macros = vstring_export(eod_macros);
    mp->unk_macros = vstring_export(unk_macros);

    return (ret == 8 ? 1 : -1);
}

/* milter_macros_create - create and initialize macros structure */

MILTER_MACROS *milter_macros_create(const char *conn_macros,
				            const char *helo_macros,
				            const char *mail_macros,
				            const char *rcpt_macros,
				            const char *data_macros,
				            const char *eoh_macros,
				            const char *eod_macros,
				            const char *unk_macros)
{
    MILTER_MACROS *mp;

    mp = (MILTER_MACROS *) mymalloc(sizeof(*mp));
    mp->conn_macros = mystrdup(conn_macros);
    mp->helo_macros = mystrdup(helo_macros);
    mp->mail_macros = mystrdup(mail_macros);
    mp->rcpt_macros = mystrdup(rcpt_macros);
    mp->data_macros = mystrdup(data_macros);
    mp->eoh_macros = mystrdup(eoh_macros);
    mp->eod_macros = mystrdup(eod_macros);
    mp->unk_macros = mystrdup(unk_macros);

    return (mp);
}

/* milter_macros_alloc - allocate macros structure with simple initialization */

MILTER_MACROS *milter_macros_alloc(int mode)
{
    MILTER_MACROS *mp;

    /*
     * This macro was originally in milter.h, but no-one else needed it.
     */
#define milter_macros_init(mp, expr) do { \
	MILTER_MACROS *__mp = (mp); \
	char *__expr = (expr); \
	__mp->conn_macros = __expr; \
	__mp->helo_macros = __expr; \
	__mp->mail_macros = __expr; \
	__mp->rcpt_macros = __expr; \
	__mp->data_macros = __expr; \
	__mp->eoh_macros = __expr; \
	__mp->eod_macros = __expr; \
	__mp->unk_macros = __expr; \
    } while (0)

    mp = (MILTER_MACROS *) mymalloc(sizeof(*mp));
    switch (mode) {
    case MILTER_MACROS_ALLOC_ZERO:
	milter_macros_init(mp, 0);
	break;
    case MILTER_MACROS_ALLOC_EMPTY:
	milter_macros_init(mp, mystrdup(""));
	break;
    default:
	msg_panic("milter_macros_alloc: unknown mode %d", mode);
    }
    return (mp);
}

/* milter_macros_free - destroy memory for MILTER_MACROS structure */

void    milter_macros_free(MILTER_MACROS *mp)
{

    /*
     * This macro was originally in milter.h, but no-one else needed it.
     */
#define milter_macros_wipe(mp) do { \
	MILTER_MACROS *__mp = mp; \
	if (__mp->conn_macros) \
	    myfree(__mp->conn_macros); \
	if (__mp->helo_macros) \
	    myfree(__mp->helo_macros); \
	if (__mp->mail_macros) \
	    myfree(__mp->mail_macros); \
	if (__mp->rcpt_macros) \
	    myfree(__mp->rcpt_macros); \
	if (__mp->data_macros) \
	    myfree(__mp->data_macros); \
	if (__mp->eoh_macros) \
	    myfree(__mp->eoh_macros); \
	if (__mp->eod_macros) \
	    myfree(__mp->eod_macros); \
	if (__mp->unk_macros) \
	    myfree(__mp->unk_macros); \
    } while (0)

    milter_macros_wipe(mp);
    myfree((void *) mp);
}
