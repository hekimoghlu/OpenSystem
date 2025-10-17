/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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

#	$OpenBSD: Makefile.inc,v 1.17 2024/01/28 01:07:26 deraadt Exp $
#	$NetBSD: Makefile.inc,v 1.8 1997/04/06 08:39:38 cgd Exp $

BINDIR=			/usr/mdec

# For descriptions of regions available to bootstrap programs, see
# section 3.4.1.2 (pp. III 3-14 - III 3-18) of the second edition of
# the Alpha AXP Architecture Reference Manual.

# "Region 1 start"
PRIMARY_LOAD_ADDRESS=	20000000
# "Region 1 start" + 32k
SECONDARY_LOAD_ADDRESS=	20004000
# "Region 1 start" + 256k
HEAP_LIMIT=		20040000

CPPFLAGS+=		-DPRIMARY_LOAD_ADDRESS="0x${PRIMARY_LOAD_ADDRESS}"
CPPFLAGS+=		-DSECONDARY_LOAD_ADDRESS="0x${SECONDARY_LOAD_ADDRESS}"

CPPFLAGS+=		-D_STANDALONE

EXTRACFLAGS+=		-DHEAP_LIMIT="0x${HEAP_LIMIT}"

CFLAGS+=		-fno-stack-protector -fno-builtin-printf \
			-fno-builtin-vprintf -fno-builtin-putchar \
			-fno-builtin-puts -fno-builtin-exit -fno-stack-protector

OBJCOPY?=		objcopy
STRIP?=			strip
