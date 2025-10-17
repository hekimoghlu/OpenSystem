/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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

#	$OpenBSD: Makefile.inc,v 1.14 2022/10/12 10:48:33 kn Exp $
#	$NetBSD: Makefile.inc,v 1.1 1996/09/30 16:34:59 ws Exp $

.if !defined(__stand_makefile_inc)
__stand_makefile_inc=1

S=$(.CURDIR)/../../../$(R)

.if !make(clean) && !make(cleandir) && !make(includes) && !make(libdep) && \
    !make(sadep) && !make(salibdir) && !make(obj) && !defined(NOMACHINE)
.BEGIN:
	@([ -h machine ] || ln -s $(S)/arch/$(MACHINE)/include machine)
.endif

#
EXTRACFLAGS=	-msoft-float
ENTRY?=	_start

INCLUDES+=	-I. -I$(.OBJDIR) -I$(.CURDIR)/.. -I$(S)/arch -I$(S)
INCLUDES+=	-I$(S)/lib/libsa
DEFS+=		-DSTANDALONE
CFLAGS+=	$(INCLUDES) $(DEFS) $(EXTRACFLAGS)
CFLAGS+=	-fno-stack-protector
CFLAGS+=	-fno-builtin -fno-pie
AFLAGS+=	-fno-pie
LDFLAGS?=	-X -N -Ttext $(RELOC) -e $(ENTRY)

cleandir:
	rm -rf lib machine

.endif
