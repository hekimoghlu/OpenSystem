/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
 *	@(#)inet.h	8.1 (Berkeley) 6/2/93
 *	$Id: inet.h,v 1.10 2006/02/01 18:09:47 majka Exp $
 */

#ifndef _ARPA_INET_H_
#define	_ARPA_INET_H_

/* External definitions for functions in inet(3), addr2ascii(3) */

#include <_bounds.h>
#include <sys/cdefs.h>
#include <sys/_types.h>
#include <stdint.h>		/* uint32_t uint16_t */
#include <machine/endian.h>	/* htonl() and family if (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */
#include <sys/_endian.h>	/* htonl() and family if (_POSIX_C_SOURCE && !_DARWIN_C_SOURCE) */
#include <netinet/in.h>		/* in_addr */

_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS

in_addr_t	 inet_addr(const char *);
char		*_LIBC_CSTR  inet_ntoa(struct in_addr);
const char	*inet_ntop(int, const void *, char *_LIBC_COUNT(__size), socklen_t __size);
int		 inet_pton(int, const char *, void *);

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
int		 ascii2addr(int, const char *, void *);
char *_LIBC_CSTR	addr2ascii(int, const void *_LIBC_SIZE(__size), int __size, char *_LIBC_UNSAFE_INDEXABLE);
int		 inet_aton(const char *, struct in_addr *);
in_addr_t	 inet_lnaof(struct in_addr);
struct in_addr	 inet_makeaddr(in_addr_t, in_addr_t);
in_addr_t	 inet_netof(struct in_addr);
in_addr_t	 inet_network(const char *);
char *_LIBC_CSTR	inet_net_ntop(int, const void *, int, char *_LIBC_COUNT(__size), __darwin_size_t __size);
int		 inet_net_pton(int, const char *, void *_LIBC_SIZE(__size), __darwin_size_t __size);
char *_LIBC_CSTR	inet_neta(in_addr_t, char *_LIBC_COUNT(__size), __darwin_size_t __size);
unsigned int	 inet_nsap_addr(const char *, unsigned char *_LIBC_COUNT(__maxlen), int __maxlen);
char *_LIBC_CSTR	inet_nsap_ntoa(int __binlen, const unsigned char *_LIBC_COUNT(__binlen), char *_LIBC_COUNT_OR_NULL(2 + __binlen*2 + __binlen/2 + 1));
#endif /* (_POSIX_C_SOURCE && !_DARWIN_C_SOURCE) */

__END_DECLS

#endif /* !_ARPA_INET_H_ */
