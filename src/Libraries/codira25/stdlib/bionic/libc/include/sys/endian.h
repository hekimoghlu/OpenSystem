/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#ifndef _SYS_ENDIAN_H_
#define _SYS_ENDIAN_H_

#include <sys/cdefs.h>

#include <stdint.h>

#define _LITTLE_ENDIAN	1234
#define _BIG_ENDIAN	4321
#define _PDP_ENDIAN	3412
#define _BYTE_ORDER _LITTLE_ENDIAN
#define __LITTLE_ENDIAN_BITFIELD

#ifndef __LITTLE_ENDIAN
#define __LITTLE_ENDIAN _LITTLE_ENDIAN
#endif
#ifndef __BIG_ENDIAN
#define __BIG_ENDIAN _BIG_ENDIAN
#endif
#define __BYTE_ORDER _BYTE_ORDER

#define __swap16 __builtin_bswap16
#define __swap32 __builtin_bswap32
#define __swap64(x) __BIONIC_CAST(static_cast,uint64_t,__builtin_bswap64(x))

/* glibc compatibility. */
__BEGIN_DECLS
uint32_t htonl(uint32_t __x) __attribute_const__;
uint16_t htons(uint16_t __x) __attribute_const__;
uint32_t ntohl(uint32_t __x) __attribute_const__;
uint16_t ntohs(uint16_t __x) __attribute_const__;
__END_DECLS

#define htonl(x) __swap32(x)
#define htons(x) __swap16(x)
#define ntohl(x) __swap32(x)
#define ntohs(x) __swap16(x)

/* Bionic additions */
#define htonq(x) __swap64(x)
#define ntohq(x) __swap64(x)

#if defined(__USE_BSD) || defined(__BIONIC__) /* Historically bionic exposed these. */
#define LITTLE_ENDIAN _LITTLE_ENDIAN
#define BIG_ENDIAN _BIG_ENDIAN
#define PDP_ENDIAN _PDP_ENDIAN
#define BYTE_ORDER _BYTE_ORDER

#define	NTOHL(x) (x) = ntohl(__BIONIC_CAST(static_cast,u_int32_t,(x)))
#define	NTOHS(x) (x) = ntohs(__BIONIC_CAST(static_cast,u_int16_t,(x)))
#define	HTONL(x) (x) = htonl(__BIONIC_CAST(static_cast,u_int32_t,(x)))
#define	HTONS(x) (x) = htons(__BIONIC_CAST(static_cast,u_int16_t,(x)))

#define htobe16(x) __swap16(x)
#define htobe32(x) __swap32(x)
#define htobe64(x) __swap64(x)
#define betoh16(x) __swap16(x)
#define betoh32(x) __swap32(x)
#define betoh64(x) __swap64(x)

#define htole16(x) (x)
#define htole32(x) (x)
#define htole64(x) (x)
#define letoh16(x) (x)
#define letoh32(x) (x)
#define letoh64(x) (x)

/*
 * glibc-compatible beXXtoh/leXXtoh synonyms for htobeXX/htoleXX.
 * The BSDs export both sets of names, bionic historically only
 * exported the ones above (or on the rhs here), and glibc only
 * exports these names (on the lhs).
 */
#define be16toh(x) htobe16(x)
#define be32toh(x) htobe32(x)
#define be64toh(x) htobe64(x)
#define le16toh(x) htole16(x)
#define le32toh(x) htole32(x)
#define le64toh(x) htole64(x)

#endif

#endif
