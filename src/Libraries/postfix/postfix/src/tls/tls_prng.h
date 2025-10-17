/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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

#ifndef _TLS_PRNG_SRC_H_INCLUDED_
#define _TLS_PRNG_SRC_H_INCLUDED_

/*++
/* NAME
/*	tls_prng_src 3h
/* SUMMARY
/*	OpenSSL PRNG maintenance routines
/* SYNOPSIS
/*	#include <tls_prng_src.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
typedef struct TLS_PRNG_SRC {
    int     fd;				/* file handle */
    char   *name;			/* resource name */
    int     timeout;			/* time limit of applicable */
} TLS_PRNG_SRC;

extern TLS_PRNG_SRC *tls_prng_egd_open(const char *, int);
extern ssize_t tls_prng_egd_read(TLS_PRNG_SRC *, size_t);
extern int tls_prng_egd_close(TLS_PRNG_SRC *);

extern TLS_PRNG_SRC *tls_prng_dev_open(const char *, int);
extern ssize_t tls_prng_dev_read(TLS_PRNG_SRC *, size_t);
extern int tls_prng_dev_close(TLS_PRNG_SRC *);

extern TLS_PRNG_SRC *tls_prng_file_open(const char *, int);
extern ssize_t tls_prng_file_read(TLS_PRNG_SRC *, size_t);
extern int tls_prng_file_close(TLS_PRNG_SRC *);

extern TLS_PRNG_SRC *tls_prng_exch_open(const char *);
extern void tls_prng_exch_update(TLS_PRNG_SRC *);
extern void tls_prng_exch_close(TLS_PRNG_SRC *);

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
