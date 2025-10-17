/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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

#	$OpenBSD: Makefile.inc,v 1.3 2014/07/16 20:08:25 okan Exp $
#	$NetBSD: Makefile.inc,v 1.5 1996/09/30 16:01:18 ws Exp $
#
#	NOTE: $S must correspond to the top of the 'sys' tree
SADIR=	$S/lib/libsa

SADST?=	lib/sa
SAREL?=	../../
SALIB?=	${SADST}/libsa.a

SAMAKE= \
	cd ${SADST} && ${MAKE} -f ${SAREL}${SADIR}/Makefile \
	  CC="${CC}" XCFLAGS="${CFLAGS}" \
	  AS='${AS}' XAFLAGS='${AFLAGS}' \
	  LD='${LD}' \
	  SAREL="${SAREL}" \
	  SADIR="${SADIR}" \
	  EXTRACFLAGS="${EXTRACFLAGS}"
.if defined(NO_NET)
SAMAKE+= NO_NET=""
.endif
.if defined(SA_ZLIB)
SAMAKE+= SA_ZLIB=""
.endif

${SALIB}:		.NOTMAIN __always_make_salib
	@echo making sure the stand-alone library is up to date...
	@${SAMAKE} libsa.a

clean:	 		.NOTMAIN __always_make_salib
	@echo cleaning the stand-alone library objects
	@${SAMAKE} clean

depend:			.NOTMAIN __always_make_salib
	@echo depending the standalone library objects
	@${SAMAKE} depend

__always_make_salib:	.NOTMAIN
	@([ -d ${SADST} ] || mkdir -p ${SADST})
