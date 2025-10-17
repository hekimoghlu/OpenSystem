/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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

#***************************************************************************
#                                  _   _ ____  _
#  Project                     ___| | | |  _ \| |
#                             / __| | | | |_) | |
#                            | (__| |_| |  _ <| |___
#                             \___|\___/|_| \_\_____|
#
# Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution. The terms
# are also available at https://curl.se/docs/copyright.html.
#
# You may opt to use, copy, modify, merge, publish, distribute and/or sell
# copies of the Software, and permit persons to whom the Software is
# furnished to do so, under the terms of the COPYING file.
#
# This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
# KIND, either express or implied.
#
# SPDX-License-Identifier: curl
#
###########################################################################

noinst_PROGRAMS = getpart resolve rtspd sockfilt sws tftpd fake_ntlm    \
 socksd disabled mqttd

CURLX_SRCS = \
 ../../lib/mprintf.c \
 ../../lib/nonblock.c \
 ../../lib/strtoofft.c \
 ../../lib/warnless.c \
 ../../lib/timediff.c \
 ../../lib/dynbuf.c \
 ../../lib/strdup.c \
 ../../lib/strcase.c \
 ../../lib/curl_multibyte.c

CURLX_HDRS = \
 ../../lib/curlx.h \
 ../../lib/nonblock.h \
 ../../lib/strtoofft.h \
 ../../lib/warnless.h \
 ../../lib/timediff.h \
 ../../lib/curl_ctype.h \
 ../../lib/dynbuf.h \
 ../../lib/strdup.h \
 ../../lib/curl_multibyte.h

USEFUL = \
 getpart.c \
 getpart.h \
 server_setup.h \
 ../../lib/base64.c \
 ../../lib/curl_base64.h \
 ../../lib/memdebug.c \
 ../../lib/memdebug.h

UTIL = \
 util.c \
 util.h

getpart_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) \
 testpart.c
getpart_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
getpart_CFLAGS = $(AM_CFLAGS)

resolve_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) $(UTIL) \
 resolve.c
resolve_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
resolve_CFLAGS = $(AM_CFLAGS)

rtspd_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) $(UTIL) \
 server_sockaddr.h \
 rtspd.c
rtspd_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
rtspd_CFLAGS = $(AM_CFLAGS)

sockfilt_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) $(UTIL) \
 server_sockaddr.h \
 sockfilt.c \
 ../../lib/inet_pton.c
sockfilt_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
sockfilt_CFLAGS = $(AM_CFLAGS)

socksd_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) $(UTIL) \
 server_sockaddr.h socksd.c \
 ../../lib/inet_pton.c
socksd_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
socksd_CFLAGS = $(AM_CFLAGS)

mqttd_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) $(UTIL) \
 server_sockaddr.h mqttd.c \
 ../../lib/inet_pton.c
mqttd_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
mqttd_CFLAGS = $(AM_CFLAGS)

sws_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) $(UTIL) \
 server_sockaddr.h \
 sws.c \
 ../../lib/inet_pton.c
sws_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
sws_CFLAGS = $(AM_CFLAGS)

tftpd_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) $(UTIL) \
 server_sockaddr.h \
 tftpd.c \
 tftp.h
tftpd_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
tftpd_CFLAGS = $(AM_CFLAGS)

fake_ntlm_SOURCES = $(CURLX_SRCS) $(CURLX_HDRS) $(USEFUL) $(UTIL) \
 fake_ntlm.c
fake_ntlm_LDADD = @CURL_NETWORK_AND_TIME_LIBS@
fake_ntlm_CFLAGS = $(AM_CFLAGS)

disabled_SOURCES = disabled.c
