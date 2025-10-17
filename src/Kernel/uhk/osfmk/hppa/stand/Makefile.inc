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

#	$OpenBSD: Makefile.inc,v 1.22 2017/07/25 13:32:14 robert Exp $

CFLAGS=${DEBUG} -Oz -Wall -Werror
CPPFLAGS+=-I${S} -I. -I${.CURDIR}
SACFLAGS=-nostdinc -fno-builtin -D_STANDALONE -I${STANDIR}/libsa
SACFLAGS+=-mdisable-fpregs -mfast-indirect-calls -mpa-risc-1-1
SACFLAGS+=-fno-stack-protector
#DEBUGFLAGS=-DDEBUG
#DEBUGFLAGS+=-DPDCDEBUG
#DEBUGFLAGS+=-DLIFDEBUG
#DEBUGFLAGS+=-DEXEC_DEBUG
#DEBUGFLAGS+=-DALLOC_TRACE
LINKADDR=0xf80000
LOADADDR=0xf80000
HEAP_LIMIT=0xff8000
CLEANFILES+=	machine

.if !make(clean) && !make(cleandir) && !make(includes) && !make(libdep) && \
    !make(sadep) && !make(salibdir) && !make(obj) && \
    (${MACHINE} == "hppa" && !(defined(PROG) && ${PROG} == "mkboot"))
.BEGIN:
	@([ -h machine ] || ln -s $(S)/arch/hppa/include machine)
.endif

.if exists(${STANDIR}/libsa/libsa.a)
LIBSA=${STANDIR}/libsa/libsa.a
.else
LIBSA=${STANDIR}/libsa/${__objdir}/libsa.a
.endif
.if exists(${STANDIR}/libz/libz.a)
LIBZ=${STANDIR}/libz/libz.a
.else
LIBZ=${STANDIR}/libz/${__objdir}/libz.a
.endif
.if exists(${STANDIR}/mkboot/mkboot)
MKBOOT=${STANDIR}/mkboot/mkboot
.else
MKBOOT=${STANDIR}/mkboot/${__objdir}/mkboot
.endif

BINDIR=	/usr/mdec
