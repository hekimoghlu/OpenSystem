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
#define IDIGIT   (1 <<  0)
#define IALNUM   (1 <<  1)
#define IBLANK   (1 <<  2)
#define INBLANK  (1 <<  3)
#define ITOK     (1 <<  4)
#define ISEP     (1 <<  5)
#define IALPHA   (1 <<  6)
#define IIDENT   (1 <<  7)
#define IUSER    (1 <<  8)
#define ICNTRL   (1 <<  9)
#define IWORD    (1 << 10)
#define ISPECIAL (1 << 11)
#define IMETA    (1 << 12)
#define IWSEP    (1 << 13)
#define INULL    (1 << 14)
#define IPATTERN (1 << 15)
#define zistype(X,Y) (typtab[STOUC(X)] & Y)
#define idigit(X) zistype(X,IDIGIT)
#define ialnum(X) zistype(X,IALNUM)
#define iblank(X) zistype(X,IBLANK)	/* blank, not including \n */
#define inblank(X) zistype(X,INBLANK)	/* blank or \n */
#define itok(X) zistype(X,ITOK)
#define isep(X) zistype(X,ISEP)
#define ialpha(X) zistype(X,IALPHA)
#define iident(X) zistype(X,IIDENT)
#define iuser(X) zistype(X,IUSER)	/* username char */
#define icntrl(X) zistype(X,ICNTRL)
#define iword(X) zistype(X,IWORD)
#define ispecial(X) zistype(X,ISPECIAL)
#define imeta(X) zistype(X,IMETA)
#define iwsep(X) zistype(X,IWSEP)
#define inull(X) zistype(X,INULL)
#define ipattern(X) zistype(X,IPATTERN)

/*
 * Bit flags for typtab_flags --- preserved after
 * shell initialisation.
 */
#define ZTF_INIT     (0x0001) /* One-off initialisation done */
#define ZTF_INTERACT (0x0002) /* Shell interactive and reading from stdin */
#define ZTF_SP_COMMA (0x0004) /* Treat comma as a special characters */
#define ZTF_BANGCHAR (0x0008) /* Treat bangchar as a special character */

#ifdef MULTIBYTE_SUPPORT
#define WC_ZISTYPE(X,Y) wcsitype((X),(Y))
# ifdef ENABLE_UNICODE9
#  define WC_ISPRINT(X)	u9_iswprint(X)
# else
#  define WC_ISPRINT(X)	iswprint(X)
# endif
#else
#define WC_ZISTYPE(X,Y)	zistype((X),(Y))
#define WC_ISPRINT(X)	isprint(X)
#endif

#if defined(__APPLE__) && defined(BROKEN_ISPRINT)
#define ZISPRINT(c)  isprint_ascii(c)
#else
#define ZISPRINT(c)  isprint(c)
#endif
