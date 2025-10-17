/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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

#	$OpenBSD: Makefile.inc,v 1.16 2024/01/11 01:45:58 djm Exp $

.include <bsd.own.mk>
.include <bsd.obj.mk>

# User-settable options
UNITTEST_FAST?= no	# Skip slow tests (e.g. less intensive fuzzing).
UNITTEST_SLOW?= no	# Include slower tests (e.g. more intensive fuzzing).
UNITTEST_VERBOSE?= no	# Verbose test output (inc. per-test names).

MALLOC_OPTIONS?=	CFGJRSUX
TEST_ENV?=		MALLOC_OPTIONS=${MALLOC_OPTIONS}

# XXX detect from ssh binary?
OPENSSL?=	yes
DSAKEY?=	yes

.if (${DSAKEY:L} == "yes")
CFLAGS+=	-DWITH_DSA
.endif

.if (${OPENSSL:L} == "yes")
CFLAGS+=	-DWITH_OPENSSL
.endif

# enable warnings
WARNINGS=Yes

DEBUG=-g
CFLAGS+=	-fstack-protector-all
CDIAGFLAGS=	-Wall
CDIAGFLAGS+=	-Wextra
CDIAGFLAGS+=	-Werror
CDIAGFLAGS+=	-Wchar-subscripts
CDIAGFLAGS+=	-Wcomment
CDIAGFLAGS+=	-Wformat
CDIAGFLAGS+=	-Wformat-security
CDIAGFLAGS+=	-Wimplicit
CDIAGFLAGS+=	-Winline
CDIAGFLAGS+=	-Wmissing-declarations
CDIAGFLAGS+=	-Wmissing-prototypes
CDIAGFLAGS+=	-Wparentheses
CDIAGFLAGS+=	-Wpointer-arith
CDIAGFLAGS+=	-Wreturn-type
CDIAGFLAGS+=	-Wshadow
CDIAGFLAGS+=	-Wsign-compare
CDIAGFLAGS+=	-Wstrict-aliasing
CDIAGFLAGS+=	-Wstrict-prototypes
CDIAGFLAGS+=	-Wswitch
CDIAGFLAGS+=	-Wtrigraphs
CDIAGFLAGS+=	-Wuninitialized
CDIAGFLAGS+=	-Wunused
CDIAGFLAGS+=	-Wno-unused-parameter
.if ${COMPILER_VERSION:L} != "gcc3"
CDIAGFLAGS+=	-Wold-style-definition
.endif

SSHREL=../../../../../usr.bin/ssh

CFLAGS+=-I${.CURDIR}/../test_helper -I${.CURDIR}/${SSHREL}

.if exists(${.CURDIR}/../test_helper/${__objdir})
LDADD+=-L${.CURDIR}/../test_helper/${__objdir} -ltest_helper
DPADD+=${.CURDIR}/../test_helper/${__objdir}/libtest_helper.a
.else
LDADD+=-L${.CURDIR}/../test_helper -ltest_helper
DPADD+=${.CURDIR}/../test_helper/libtest_helper.a
.endif

.PATH: ${.CURDIR}/${SSHREL}

LDADD+= -lutil
DPADD+= ${LIBUTIL}

.if (${OPENSSL:L} == "yes")
LDADD+= -lcrypto
DPADD+= ${LIBCRYPTO}
.endif

LDADD+= -lfido2 -lcbor -lusbhid
DPADD+= ${LIBFIDO2} ${LIBCBOR} ${LIBUSBHID}

UNITTEST_ARGS?=

.if (${UNITTEST_VERBOSE:L} != "no")
UNITTEST_ARGS+= -v
.endif
.if (${UNITTEST_FAST:L} != "no")
UNITTEST_ARGS+= -f
.elif (${UNITTEST_SLOW:L} != "no")
UNITTEST_ARGS+= -F
.endif
