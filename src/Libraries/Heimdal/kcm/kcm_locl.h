/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
/*
 * $Id$
 */

#ifndef __KCM_LOCL_H__
#define __KCM_LOCL_H__

#include "headers.h"

#include <kcm.h>

#define KCM_LOG_REQUEST(_context, _client, _opcode)	do { \
    kcm_log(1, "%s request by process %d/uid %d", \
	    kcm_op2string(_opcode), (_client)->pid, (_client)->uid); \
    } while (0)

#define KCM_LOG_REQUEST_NAME(_context, _client, _opcode, _name)	do { \
    kcm_log(1, "%s request for cache %s by process %d/uid %d", \
	    kcm_op2string(_opcode), (_name), (_client)->pid, (_client)->uid); \
    } while (0)

/* Cache management */

#define KCM_FLAGS_USE_KEYTAB		0x0001
#define KCM_FLAGS_USE_PASSWORD		0x0002

#define KCM_FLAGS_RENEWABLE		0x0010
#define KCM_FLAGS_OWNER_IS_SYSTEM	0x0020

#define KCM_MASK_KEY_PRESENT (KCM_FLAGS_USE_KEYTAB|KCM_FLAGS_USE_PASSWORD)


struct kcm_creds;

struct kcm_default_cache {
    uid_t uid;
    pid_t session; /* really au_asid_t */
    char *name;
    struct kcm_default_cache *next;
};

extern struct kcm_default_cache *default_caches;

struct kcm_creds {
    kcmuuid_t uuid;
    krb5_creds cred;
    struct kcm_creds *next;
};

struct kcm_ccache_data {
    char *name;
    kcmuuid_t uuid;
    long holdcount;
    unsigned refcnt;
    uint16_t flags;
    uid_t uid;
    pid_t session; /* really au_asid_t */
    krb5_principal client; /* primary client principal */
    krb5_principal server; /* primary server principal (TGS if NULL) */
    struct kcm_creds *creds;
    krb5_deltat tkt_life;
    krb5_deltat renew_life;
    int32_t kdc_offset;
    heim_event_t renew_event;
    time_t renew_time;
    heim_event_t expire_event;
    krb5_deltat expire;
    time_t next_refresh_time;
    /* key */
    krb5_keytab keytab;
    char *password;

    HEIMDAL_MUTEX mutex;
    TAILQ_ENTRY(kcm_ccache_data) members;
};

#define KCM_ASSERT_VALID(_ccache)		do { \
    if ((_ccache)->refcnt == 0) \
	krb5_abortx(context, "kcm_free_ccache_data: ccache refcnt == 0"); \
    } while (0)

typedef struct kcm_ccache_data *kcm_ccache;

/* Request format is  LENGTH | MAJOR | MINOR | OPERATION | request */
/* Response format is LENGTH | STATUS | response */

typedef enum {
    IAKERB_NOT_CHECKED = 0,
    IAKERB_ACCESS_DENIED = 1,
    IAKERB_ACCESS_GRANTED = 2
} iakerb_access_status;

typedef struct kcm_client {
    pid_t pid;
    uid_t uid;
    pid_t session;
    iakerb_access_status iakerb_access;
    audit_token_t audit_token;
    char execpath[MAXPATHLEN];
} kcm_client;

#define CLIENT_IS_ROOT(client) ((client)->uid == 0)

/* Dispatch table */
/* passed in OPERATION | ... ; returns STATUS | ... */
typedef krb5_error_code (*kcm_method)(krb5_context, kcm_client *, kcm_operation, krb5_storage *, krb5_storage *);

struct kcm_op {
    const char *name;
    kcm_method method;
};

#define DEFAULT_LOG_DEST    "0/SYSLOG:DEBUG:DAEMON"
#define _PATH_KCM_CONF	    SYSCONFDIR "/kcm.conf"

extern krb5_context kcm_context;
extern char *socket_path;
extern char *door_path;
extern size_t max_request;
extern sig_atomic_t exit_flag;
extern int max_num_requests;
extern int kcm_timeout;
#ifdef SUPPORT_DETACH
extern int detach_from_console;
#endif
extern int launchd_flag;
extern int disallow_getting_krbtgt;
extern int kcm_data_changed;
extern int use_uid_matching;
extern int disable_ntlm_reflection_detection;

#if 0
extern const krb5_cc_ops krb5_kcmss_ops;
#endif

#include <kcm-protos.h>

#if __APPLE__
#include <sys/codesign.h>
#include <Security/Security.h>
#endif

#define CFRELEASE_NULL(x) do { if (x) { CFRelease(x); x = NULL; } } while(0)

#endif /* __KCM_LOCL_H__ */

