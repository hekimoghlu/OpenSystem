/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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

#ifndef _BOUNCE_LOG_H_INCLUDED_
#define _BOUNCE_LOG_H_INCLUDED_

/*++
/* NAME
/*	bounce_log 3h
/* SUMMARY
/*	bounce file reader
/* SYNOPSIS
/*	#include <bounce_log.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstream.h>
#include <vstring.h>

 /*
  * Global library.
  */
#include <recipient_list.h>
#include <rcpt_buf.h>
#include <dsn_buf.h>

 /*
  * External interface.
  */
typedef struct {
    VSTREAM *fp;			/* open file */
    VSTRING *buf;			/* I/O buffer */
    char    *compat_status;		/* old logfile compatibility */
    char    *compat_action;		/* old logfile compatibility */
} BOUNCE_LOG;

extern BOUNCE_LOG *bounce_log_open(const char *, const char *, int, mode_t);
extern BOUNCE_LOG *bounce_log_read(BOUNCE_LOG *, RCPT_BUF *, DSN_BUF *);
extern BOUNCE_LOG *bounce_log_delrcpt(BOUNCE_LOG *);
extern int bounce_log_close(BOUNCE_LOG *);

#define bounce_log_rewind(bp) vstream_fseek((bp)->fp, 0L, SEEK_SET)

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
