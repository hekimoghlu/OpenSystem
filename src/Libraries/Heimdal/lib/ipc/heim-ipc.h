/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#ifndef __HEIM_IPC_H
#define __HEIM_IPC_H 1

#include <krb5-types.h>
#include <asn1-common.h>
#include <bsm/libbsm.h>
#include <CoreFoundation/CoreFoundation.h>

typedef struct heim_ipc *heim_ipc;
typedef struct heim_sipc *heim_sipc;
typedef struct heim_icred *heim_icred;
typedef struct heim_isemaphore *heim_isemaphore;
typedef struct heim_base_data heim_idata;
typedef struct heim_sipc_call *heim_sipc_call;

/* common */

void
heim_ipc_free_cred(heim_icred);

uid_t
heim_ipc_cred_get_uid(heim_icred);

gid_t
heim_ipc_cred_get_gid(heim_icred);

pid_t
heim_ipc_cred_get_pid(heim_icred);

pid_t
heim_ipc_cred_get_session(heim_icred);

audit_token_t
heim_ipc_cred_get_audit_token(heim_icred);

struct sockaddr *
heim_ipc_cred_get_client_address(heim_icred cred, krb5_socklen_t *sa_size);

struct sockaddr *
heim_ipc_cred_get_server_address(heim_icred cred, krb5_socklen_t *sa_size);


void
heim_ipc_main(void)
    __attribute__((__noreturn__));

heim_isemaphore
heim_ipc_semaphore_create(long);

long
heim_ipc_semaphore_wait(heim_isemaphore, time_t);

long
heim_ipc_semaphore_signal(heim_isemaphore);

void
heim_ipc_semaphore_release(heim_isemaphore);

#define HEIM_IPC_WAIT_FOREVER ((time_t)-1)

void
heim_ipc_free_data(heim_idata *);

/* client */

int
heim_ipc_init_context(const char *, heim_ipc *);

void
heim_ipc_free_context(heim_ipc);

int
heim_ipc_call(heim_ipc, const heim_idata *, heim_idata *, heim_icred *);

int
heim_ipc_async(heim_ipc, const heim_idata *, void *, void (*func)(void *, int, heim_idata *, heim_icred));

/* server */

#define HEIM_SIPC_TYPE_IPC		1
#define HEIM_SIPC_TYPE_UINT32		2
#define HEIM_SIPC_TYPE_HTTP		4
#define HEIM_SIPC_TYPE_ONE_REQUEST	8

typedef void
(*heim_ipc_complete)(heim_sipc_call, int, heim_idata *);

typedef void
(*heim_ipc_callback)(void *, const heim_idata *,
		     const heim_icred, heim_ipc_complete, heim_sipc_call);


int
heim_sipc_launchd_mach_init(const char *, heim_ipc_callback,
			    void *, heim_sipc *);

int
heim_sipc_stream_listener(int, int, heim_ipc_callback,
			  void *, heim_sipc *);

int
heim_sipc_service_dgram(int, int, heim_ipc_callback,
			void *, heim_sipc *);

int
heim_sipc_service_unix(const char *, heim_ipc_callback,
		       void *, heim_sipc *);


void
heim_sipc_timeout(time_t);

void
heim_sipc_set_timeout_handler(void (*)(void));

void
heim_sipc_free_context(heim_sipc);

typedef struct heim_event_data *heim_event_t;

typedef void (*heim_ipc_event_callback_t)(heim_event_t, void *);
typedef void (*heim_ipc_event_final_t)(void *);

heim_event_t
heim_ipc_event_create_f(heim_ipc_event_callback_t, void *);

heim_event_t
heim_ipc_event_cf_create_f(heim_ipc_event_callback_t cb, CFTypeRef ctx);

int
heim_ipc_event_set_final(heim_event_t, heim_ipc_event_final_t);

int
heim_ipc_event_set_time(heim_event_t, time_t);

void
heim_ipc_event_cancel(heim_event_t);

bool
heim_ipc_event_is_cancelled(heim_event_t e);

void
heim_ipc_event_set_final_f(heim_event_t, heim_ipc_event_final_t );

void
heim_ipc_event_free(heim_event_t);

void
heim_ipc_init_globals(void);

void
heim_ipc_resume_events(void);

void
heim_ipc_suspend_events(void);
/*
 * Signal helpers
 */

void
heim_sipc_signal_handler(int, void (*)(void *), void *);


#endif /* __HEIM_IPC_H */
