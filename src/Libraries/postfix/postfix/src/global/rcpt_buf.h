/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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

#ifndef _RCPT_BUF_H_INCLUDED_
#define _RCPT_BUF_H_INCLUDED_

/*++
/* NAME
/*	rcpt_buf 3h
/* SUMMARY
/*	recipient buffer manager
/* SYNOPSIS
/*	#include <rcpt_buf.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstream.h>
#include <vstring.h>
#include <attr.h>

 /*
  * Global library.
  */
#include <recipient_list.h>

 /*
  * External interface.
  */
typedef struct {
    RECIPIENT rcpt;			/* convenience */
    VSTRING *address;			/* final recipient */
    VSTRING *orig_addr;			/* original recipient */
    VSTRING *dsn_orcpt;			/* dsn original recipient */
    int     dsn_notify;			/* DSN notify flags */
    long    offset;			/* REC_TYPE_RCPT byte */
} RCPT_BUF;

extern RCPT_BUF *rcpb_create(void);
extern void rcpb_reset(RCPT_BUF *);
extern void rcpb_free(RCPT_BUF *);
extern int rcpb_scan(ATTR_SCAN_MASTER_FN, VSTREAM *, int, void *);

#define RECIPIENT_FROM_RCPT_BUF(buf) \
    ((buf)->rcpt.address = vstring_str((buf)->address), \
     (buf)->rcpt.orig_addr = vstring_str((buf)->orig_addr), \
     (buf)->rcpt.dsn_orcpt = vstring_str((buf)->dsn_orcpt), \
     (buf)->rcpt.dsn_notify = (buf)->dsn_notify, \
     (buf)->rcpt.offset = (buf)->offset, \
     &(buf)->rcpt)

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
