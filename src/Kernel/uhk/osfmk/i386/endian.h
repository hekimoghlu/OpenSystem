/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
 * @OSF_COPYRIGHT@
 *
 */

#ifndef _MACHINE_ENDIAN_H_
#define _MACHINE_ENDIAN_H_

/*
 * Definitions for byte order,
 * according to byte significance from low address to high.
 */
#define LITTLE_ENDIAN   1234    /* least-significant byte first (vax) */
#define BIG_ENDIAN      4321    /* most-significant byte first (IBM, net) */
#define PDP_ENDIAN      3412    /* LSB first in word, MSW first in long (pdp) */

#define BYTE_ORDER      LITTLE_ENDIAN   /* byte order on i386 */
#define ENDIAN          LITTLE

/*
 * Macros for network/external number representation conversion.
 * Use GNUC support to inline the byteswappers.
 */

#if !defined(ntohs)
static __inline__ unsigned short        ntohs(unsigned short);
static __inline__
unsigned short
ntohs(unsigned short w_int)
{
	return (unsigned short)((w_int << 8) | (w_int >> 8));
}
#endif

#if !defined(htons)
unsigned short  htons(unsigned short);
#define htons   ntohs
#endif

#if !defined(ntohl)
static __inline__ unsigned long ntohl(unsigned long);
static __inline__
unsigned long
ntohl(unsigned long value)
{
#if defined(__clang__)
	return (unsigned long)__builtin_bswap32((unsigned int)value);
#else
	unsigned long l = value;
	__asm__ volatile ("bswap %0" : "=r" (l) : "0" (l));
	return l;
#endif
}
#endif

#if !defined(htonl)
unsigned long   htonl(unsigned long);
#define htonl   ntohl
#endif

#define NTOHL(x)        (x) = ntohl((unsigned long)x)
#define NTOHS(x)        (x) = ntohs((unsigned short)x)
#define HTONL(x)        (x) = htonl((unsigned long)x)
#define HTONS(x)        (x) = htons((unsigned short)x)

#endif
