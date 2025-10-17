/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#include "config.h"

#include <sys/types.h>
#include <sys/socket.h>
#ifdef HAVE_SYS_UN_H
#include <sys/un.h>
#endif

#define __APPLE_USE_RFC_3542 1

#include <netinet/in.h>

#include <sys/poll.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef HAVE_GETPEERUCRED
#include <ucred.h>
#endif

#include <krb5-types.h>
#include <asn1-common.h>

#include <heimbase.h>
#include <base64.h>

#include <heim-ipc.h>

#define MAX_PACKET_SIZE (128 * 1024)

#if defined(__APPLE__) && defined(HAVE_GCD)
#include <mach/mach.h>
#include <servers/bootstrap.h>
#include <dispatch/dispatch.h>
#include <bsm/libbsm.h>

#ifndef __APPLE_PRIVATE__ /* awe, using private interface */
typedef boolean_t (*dispatch_mig_callback_t)(mach_msg_header_t *message, mach_msg_header_t *reply);

mach_msg_return_t
dispatch_mig_server(dispatch_source_t ds, size_t maxmsgsz, dispatch_mig_callback_t callback);
#else
#include <dispatch/private.h>
#endif

#endif


#include <roken.h>

int
_heim_ipc_create_cred(uid_t, gid_t, pid_t, pid_t, heim_icred *);

int
_heim_ipc_create_cred_with_audit_token(uid_t, gid_t, pid_t, pid_t, audit_token_t, heim_icred *);

int
_heim_ipc_create_network_cred(struct sockaddr *, krb5_socklen_t,
			      struct sockaddr *, krb5_socklen_t,
			      heim_icred *);
void
_heim_ipc_suspend_timer(void);

void
_heim_ipc_restart_timer(void);
