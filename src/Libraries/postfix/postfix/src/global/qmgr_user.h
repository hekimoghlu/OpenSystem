/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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

#ifndef _QMGR_USER_H_INCLUDED_
#define _QMGR_USER_H_INCLUDED_

/*++
/* NAME
/*	qmgr_user 3h
/* SUMMARY
/*	qmgr user interface codes
/* SYNOPSIS
/*	#include <qmgr_user.h>
/* DESCRIPTION
/* .nf

 /*
  * Global library.
  */
#include <dsn_mask.h>

 /*
  * Queue file read options. Flags 16- are reserved by qmgr.h; unfortunately
  * DSN_NOTIFY_* needs to be shifted to avoid breaking compatibility with
  * already queued mail that uses QMGR_READ_FLAG_MIXED_RCPT_OTHER.
  */
#define QMGR_READ_FLAG_NONE		0	/* No special features */
#define QMGR_READ_FLAG_MIXED_RCPT_OTHER	(1<<0)
#define QMGR_READ_FLAG_FROM_DSN(x)	((x) << 1)

#define QMGR_READ_FLAG_NOTIFY_NEVER	(DSN_NOTIFY_NEVER << 1)
#define QMGR_READ_FLAG_NOTIFY_SUCCESS	(DSN_NOTIFY_SUCCESS << 1)
#define QMGR_READ_FLAG_NOTIFY_DELAY	(DSN_NOTIFY_DELAY << 1)
#define QMGR_READ_FLAG_NOTIFY_FAILURE	(DSN_NOTIFY_FAILURE << 1)

#define QMGR_READ_FLAG_USER \
    (QMGR_READ_FLAG_NOTIFY_NEVER | QMGR_READ_FLAG_NOTIFY_SUCCESS \
    | QMGR_READ_FLAG_NOTIFY_DELAY | QMGR_READ_FLAG_NOTIFY_FAILURE \
    | QMGR_READ_FLAG_MIXED_RCPT_OTHER)

 /*
  * Backwards compatibility.
  */
#define QMGR_READ_FLAG_DEFAULT	(QMGR_READ_FLAG_MIXED_RCPT_OTHER)

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
