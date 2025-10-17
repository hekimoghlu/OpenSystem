/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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
/*-
 * Copyright (c) 2013 The NetBSD Foundation, Inc.
 * All rights reserved.
 *
 * This code is derived from software contributed to The NetBSD Foundation
 * by Christos Zoulas.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE NETBSD FOUNDATION, INC. AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef _DNS_NET_HOSTENT_H
#define _DNS_NET_HOSTENT_H

#include <stdio.h>
#include <netdb.h>
#include <stdarg.h>

/*
 * These are not being advertised because the interfaces are non-standard.
 * There are versions by linux, aix, qnx, sun, etc. Our versions are used
 * internally to provide thread safety; they mostly resemble qnx.
 */
void sethostent_r(FILE **);
struct hostent	*netbsd_gethostent_r(FILE *, struct hostent *, char *, size_t, int *);
void endhostent_r(FILE **);

/*
 * The following are internal API's and are used only for testing.
 */
struct getnamaddr {
	struct hostent *hp;
	char *buf;
	size_t buflen;
	int *he;
};

/* /etc/hosts lookup */
int _hf_gethtbyaddr(void *, void *, va_list);
int _hf_gethtbyname(void *, void *, va_list);

#ifdef YP
/* NIS lookup */
int _yp_gethtbyaddr(void *, void *, va_list);
int _yp_gethtbyname(void *, void *, va_list);
#endif

#define HENT_ARRAY(dst, anum, ptr, len) \
	do { \
		size_t _len = (anum + 1) * sizeof(*dst); \
		if (_len > len) \
			goto nospc; \
		dst = (void *)ptr; \
		ptr += _len; \
		len -= _len; \
	} while (/*CONSTCOND*/0)

#define HENT_COPY(dst, src, slen, ptr, len) \
	do { \
		if ((size_t)slen > len) \
			goto nospc; \
		memcpy(ptr, src, (size_t)slen); \
		dst = ptr; \
		ptr += slen; \
		len -= slen; \
	} while (/* CONSTCOND */0)

#define HENT_SCOPY(dst, src, ptr, len) \
	do { \
		size_t _len = strlen(src) + 1; \
		HENT_COPY(dst, src, _len, ptr, len); \
	} while (/* CONSTCOND */0)

#endif /* _DNS_NET_HOSTENT_H */
