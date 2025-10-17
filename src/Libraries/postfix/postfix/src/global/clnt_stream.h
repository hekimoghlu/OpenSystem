/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#ifndef _CLNT_STREAM_H_INCLUDED_
#define _CLNT_STREAM_H_INCLUDED_

/*++
/* NAME
/*	clnt_stream 3h
/* SUMMARY
/*	client socket maintenance
/* SYNOPSIS
/*	#include <clnt_stream.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstream.h>

 /*
  * External interface.
  */
typedef struct CLNT_STREAM CLNT_STREAM;

extern CLNT_STREAM *clnt_stream_create(const char *, const char *, int, int);
extern VSTREAM *clnt_stream_access(CLNT_STREAM *);
extern void clnt_stream_recover(CLNT_STREAM *);
extern void clnt_stream_free(CLNT_STREAM *);

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
