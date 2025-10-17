/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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

#	$OpenBSD: Makefile.inc,v 1.6 2019/11/28 00:17:12 bluhm Exp $

NOMAN=
S=		${.CURDIR}/../../../..
BOOTDIR=	${S}/arch/loongson/stand/boot

CPPFLAGS+=	${SAABI} -mno-abicalls -D_NO_ABICALLS -DCONSPEED=9600 \
		-nostdinc -D__loongson__ \
		-I${S} -I${S}/lib/libsa \
		-I${.OBJDIR} -I${BOOTDIR}

LDSCRIPT=	${.CURDIR}/ld.script
LDFLAGS+=	${SALDFLAGS} -T ${LDSCRIPT} -e __start

AFLAGS+=	${SAABI}

.PATH:		${BOOTDIR}
SRCS=		start.S
SRCS+=		conf.c cons.c dev.c devopen.c exec.c machdep.c rd.c
.PATH:		${S}/arch/loongson/loongson
SRCS+=		pmon.c pmon32.S

#### MI boot code
.PATH:		${S}/stand/boot
SRCS+=		boot.c cmd.c vars.c

#### libsa
.PATH:  	${S}/lib/libsa
SRCS+=  	hexdump.c strtoll.c

#### libkern
.PATH:		${S}/lib/libkern/arch/mips64 ${S}/lib/libkern
SRCS+=		strlcat.c strlcpy.c strlen.c
SRCS+=		udivdi3.c qdivrem.c

.if !make(clean) && !make(cleandir) && !make(includes) && !make(obj)
.BEGIN:
	@([ -h machine ] || ln -s ${.CURDIR}/../../include machine)
	@([ -h mips64 ] || ln -s ${.CURDIR}/../../../mips64/include mips64)
CLEANFILES+= machine mips64
.endif

${PROG}: ${OBJS} ${LIBSA}
	${LD} ${LDFLAGS} -o ${PROG} ${OBJS} ${LIBSA}
