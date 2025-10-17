/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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
/* crc32.h -- compute the CRC-32 of a data stream
 * Copyright (C) 1995 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef __crc32_h
#define __crc32_h       /* identifies this source module */

/* This header should be read AFTER zip.h resp. unzip.h
 * (the latter with UNZIP_INTERNAL defined...).
 */

#ifndef OF
#  define OF(a) a
#endif
#ifndef ZCONST
#  define ZCONST const
#endif

#ifdef DYNALLOC_CRCTAB
   void     free_crc_table  OF((void));
#endif
#ifndef USE_ZLIB
   ZCONST ulg near *get_crc_table  OF((void));
#endif
#if (defined(USE_ZLIB) || defined(CRC_TABLE_ONLY))
#  ifdef IZ_CRC_BE_OPTIMIZ
#    undef IZ_CRC_BE_OPTIMIZ
#  endif
#else /* !(USE_ZLIB || CRC_TABLE_ONLY) */
   ulg      crc32           OF((ulg crc, ZCONST uch *buf, extent len));
#endif /* ?(USE_ZLIB || CRC_TABLE_ONLY) */

#ifndef CRC_32_TAB
#  define CRC_32_TAB     crc_32_tab
#endif

#ifdef CRC32
#  undef CRC32
#endif
#ifdef IZ_CRC_BE_OPTIMIZ
#  define CRC32UPD(c, crctab) (crctab[((c) >> 24)] ^ ((c) << 8))
#  define CRC32(c, b, crctab) (crctab[(((int)(c) >> 24) ^ (b))] ^ ((c) << 8))
#  define REV_BE(w) (((w)>>24)+(((w)>>8)&0xff00)+ \
                    (((w)&0xff00)<<8)+(((w)&0xff)<<24))
#else
#  define CRC32UPD(c, crctab) (crctab[((int)(c)) & 0xff] ^ ((c) >> 8))
#  define CRC32(c, b, crctab) (crctab[((int)(c) ^ (b)) & 0xff] ^ ((c) >> 8))
#  define REV_BE(w) w
#endif

#endif /* !__crc32_h */
