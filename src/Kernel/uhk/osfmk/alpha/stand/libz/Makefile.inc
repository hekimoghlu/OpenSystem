/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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

#	$OpenBSD: Makefile.inc,v 1.1 1997/05/05 06:01:58 millert Exp $
#	$NetBSD: Makefile.inc,v 1.2 1997/01/23 22:29:09 cgd Exp $
#
#	Configuration variables (default values are below):
#
#	S	must be set to the top of the 'sys' tree.
#	ZDST	may be set to the location of the directory where library
#		objects are to be built.  Defaults to ${.OBJDIR}/lib/z.
#	Z_AS	may be set to 'obj' to build a object from the library's
#		object files.  (Otherwise, a library will be built.)
#		Defaults to 'library'.
#	ZMISCMAKEFLAGS
#		Miscellaneous flags to be passed to the library's Makefile when
#		building.  See library's Makefile for more details about
#		supported flags and their default values.

# Default values:
ZDST?=		${.OBJDIR}/lib/z
Z_AS?=		library

ZDIR=		$S/lib/libz
.if (${Z_AS} == "obj")
ZLIB=		${ZDST}/libz.o
ZLIB_PROF=	${ZDST}/libz.po
.else
ZLIB=		${ZDST}/libz.a
ZLIB_PROF=	${ZDST}/libz_p.a
.endif

ZMAKE= \
	cd ${ZDIR} && MAKEOBJDIR=${ZDST} ${MAKE} \
	    CC="${CC}" CFLAGS="${CFLAGS}" \
	    AS=${AS} AFLAGS="${AFLAGS}" \
	    LD=${LD} STRIP=${STRIP} \
	    MACHINE=${MACHINE} MACHINE_ARCH=${MACHINE_ARCH} \
	    ZCPPFLAGS="${CPPFLAGS:S@^-I.@-I../../.@g}" \
	    ${ZMISCMAKEFLAGS}

${ZLIB}:		.NOTMAIN __always_make_zlib
	@echo making sure the z library is up to date...
.if (${Z_AS} == "library")
	${ZMAKE} libz.a
.else
	@${ZMAKE} libz.o
.endif

${ZLIB_PROF}:		.NOTMAIN __always_make_zlib
	@echo making sure the profiled z library is up to date...
.if (${Z_AS} == "library")
	@${ZMAKE} libz_p.a
.else
	@${ZMAKE} libz.po
.endif

clean:			.NOTMAIN cleanzlib
cleanzlib:		.NOTMAIN __always_make_zlib
	@echo cleaning the z library objects
	@${ZMAKE} clean

cleandir:		.NOTMAIN cleandirzlib
cleandirzlib:		.NOTMAIN __always_make_zlib
	@echo cleandiring the z library objects
	@${ZMAKE} cleandir

depend:			.NOTMAIN dependzlib
dependzlib:		.NOTMAIN __always_make_zlib
	@echo depending the z library objects
	@${ZMAKE} depend

__always_make_zlib:	.NOTMAIN
	@mkdir -p ${ZDST}
