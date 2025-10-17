/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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
/*
 * Copyright (C) 1990 by the Massachusetts Institute of Technology
 *
 * Export of this software from the United States of America is assumed
 * to require a specific license from the United States Government.
 * It is the responsibility of any person or organization contemplating
 * export to obtain such a license before exporting.
 *
 * WITHIN THAT CONSTRAINT, permission to use, copy, modify, and
 * distribute this software and its documentation for any purpose and
 * without fee is hereby granted, provided that the above copyright
 * notice appear in all copies and that both that copyright notice and
 * this permission notice appear in supporting documentation, and that
 * the name of M.I.T. not be used in advertising or publicity pertaining
 * to distribution of the software without specific, written prior
 * permission.  M.I.T. makes no representations about the suitability of
 * this software for any purpose.  It is provided "as is" without express
 * or implied warranty.
 */

/* $Id$ */

#ifndef	__ENCRYPT__
#define	__ENCRYPT__

#define	DIR_DECRYPT		1
#define	DIR_ENCRYPT		2

#define	VALIDKEY(key)	( key[0] | key[1] | key[2] | key[3] | \
			  key[4] | key[5] | key[6] | key[7])

#define	SAMEKEY(k1, k2)	(!memcmp(k1, k2, sizeof(des_cblock)))

typedef	struct {
	short		type;
	int		length;
	unsigned char	*data;
} Session_Key;

typedef struct {
	char	*name;
	int	type;
	void	(*output) (unsigned char *, int);
	int	(*input) (int);
	void	(*init) (int);
	int	(*start) (int, int);
	int	(*is) (unsigned char *, int);
	int	(*reply) (unsigned char *, int);
	void	(*session) (Session_Key *, int);
	int	(*keyid) (int, unsigned char *, int *);
	void	(*printsub) (unsigned char *, size_t, unsigned char *, size_t);
} Encryptions;

#define	SK_DES		1	/* Matched Kerberos v5 KEYTYPE_DES */

#include "crypto-headers.h"
#ifdef HAVE_OPENSSL
#define des_new_random_key des_random_key
#endif

#include "enc-proto.h"

extern int encrypt_debug_mode;
extern int (*decrypt_input) (int);
extern void (*encrypt_output) (unsigned char *, int);
#endif
