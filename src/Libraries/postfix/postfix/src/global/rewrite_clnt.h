/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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

#ifndef _REWRITE_CLNT_H_INCLUDED_
#define _REWRITE_CLNT_H_INCLUDED_

/*++
/* NAME
/*	rewrite_clnt 3h
/* SUMMARY
/*	address rewriter client
/* SYNOPSIS
/*	#include <rewrite_clnt.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstring.h>

 /*
  * Global library.
  */
#include <mail_proto.h>			/* MAIL_ATTR_RWR_LOCAL */

 /*
  * External interface.
  */
#define REWRITE_ADDR	"rewrite"
#define REWRITE_CANON	MAIL_ATTR_RWR_LOCAL	/* backwards compatibility */

extern VSTRING *rewrite_clnt(const char *, const char *, VSTRING *);
extern VSTRING *rewrite_clnt_internal(const char *, const char *, VSTRING *);

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
