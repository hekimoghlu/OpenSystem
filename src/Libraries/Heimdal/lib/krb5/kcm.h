/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
#ifndef __KCM_H__
#define __KCM_H__

/*
 * KCM protocol definitions
 */

#define KCM_PROTOCOL_VERSION_MAJOR	2
#define KCM_PROTOCOL_VERSION_MINOR	0

typedef unsigned char kcmuuid_t[16];

typedef enum kcm_operation {
    KCM_OP_NOOP,
    KCM_OP_GET_NAME,
    KCM_OP_RESOLVE,
    KCM_OP_DEPRECATED_GEN_NEW,
    KCM_OP_INITIALIZE,
    KCM_OP_DESTROY,
    KCM_OP_STORE,
    KCM_OP_RETRIEVE,
    KCM_OP_GET_PRINCIPAL,
    KCM_OP_GET_CRED_UUID_LIST,
    KCM_OP_GET_CRED_BY_UUID,
    KCM_OP_REMOVE_CRED,
    KCM_OP_SET_FLAGS,
    KCM_OP_CHOWN,
    KCM_OP_CHMOD,
    KCM_OP_GET_INITIAL_TICKET,
    KCM_OP_GET_TICKET,
    KCM_OP_MOVE_CACHE,
    KCM_OP_GET_CACHE_UUID_LIST,
    KCM_OP_GET_CACHE_BY_UUID,
    KCM_OP_GET_DEFAULT_CACHE,
    KCM_OP_SET_DEFAULT_CACHE,
    KCM_OP_GET_KDC_OFFSET,
    KCM_OP_SET_KDC_OFFSET,
    KCM_OP_RETAIN_KCRED,
    KCM_OP_RELEASE_KCRED,
    KCM_OP_GET_UUID,
    /* NTLM operations */
    KCM_OP_ADD_NTLM_CRED,
    KCM_OP_HAVE_NTLM_CRED,
    KCM_OP_ADD_NTLM_CHALLENGE,
    KCM_OP_DO_NTLM_AUTH,
    KCM_OP_GET_NTLM_USER_LIST,
    /* SCRAM */
    KCM_OP_ADD_SCRAM_CRED,
    KCM_OP_HAVE_SCRAM_CRED,
    KCM_OP_DEL_SCRAM_CRED,
    KCM_OP_DO_SCRAM_AUTH,
    KCM_OP_GET_SCRAM_USER_LIST,
    /* GENERIC */
    KCM_OP_DESTROY_CRED,
    KCM_OP_RETAIN_CRED,
    KCM_OP_RELEASE_CRED,
    KCM_OP_CRED_LABEL_GET,
    KCM_OP_CRED_LABEL_SET,
    /* */
    KCM_OP_CHECK_NTLM_CHALLENGE,
    KCM_OP_GET_CACHE_PRINCIPAL_LIST,
    KCM_OP_MAX
} kcm_operation;

#define _PATH_KCM_SOCKET      "/var/run/.kcm_socket"
#define _PATH_KCM_DOOR      "/var/run/.kcm_door"

#define KRB5_KCM_NOTIFY_CACHE_CHANGED "com.apple.Kerberos.cache.changed"

/* notification name used on MacOS X */
#define kCCAPICacheCollectionChangedNotification "CCAPICacheCollectionChangedNotification"
#define kCCAPICCacheChangedNotification "CCAPICCacheChangedNotification"


#define KCM_STATUS_KEY			"kcm-status"
#define KCM_STATUS_ACQUIRE_START	0
#define KCM_STATUS_ACQUIRE_SUCCESS	1
#define KCM_STATUS_ACQUIRE_FAILED	2
#define KCM_STATUS_ACQUIRE_STOPPED	3


#define KCM_NTLM_FLAG_AV_GUEST 8

#endif /* __KCM_H__ */

