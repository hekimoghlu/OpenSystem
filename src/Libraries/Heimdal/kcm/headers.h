/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#ifndef __HEADERS_H__
#define __HEADERS_H__


#include <config.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_SYS_UN_H
#include <sys/un.h>
#endif
#ifdef HAVE_SYS_UCRED_H
#include <sys/ucred.h>
#endif
#ifdef HAVE_UTIL_H
#include <util.h>
#endif
#ifdef HAVE_LIBUTIL_H
#include <libutil.h>
#endif
#ifdef HAVE_GETPEERUCRED
#include <ucred.h>
#endif
#ifdef HAVE_DOOR_CREATE
#include <door.h>
#include <alloca.h>
#endif
#include <ctype.h>
#include <err.h>
#include <roken.h>
#include <getarg.h>
#include <base64.h>
#include <parse_units.h>
#include <parse_time.h>
#include <parse_bytes.h>

#ifdef __APPLE__
#include <SystemConfiguration/SCPreferences.h>
#include <dispatch/dispatch.h>
#include <notify.h>
#endif

#include <krb5.h>
#include <heim_threads.h>
#include <heim-ipc.h>

#include <gssapi.h>
#include <gssapi_spi.h>
#include <gssapi_ntlm.h>

#include <heimbase.h>

#include "crypto-headers.h"


#endif /* __HEADERS_H__ */

