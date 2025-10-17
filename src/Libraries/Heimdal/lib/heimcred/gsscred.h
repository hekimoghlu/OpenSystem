/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#import <xpc/xpc.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CFRuntime.h>
#import <os/log.h>
#import <heim-ipc.h>
#import "common.h"
#import "GSSCredHelperClient.h"
#import "gssoslog.h"

#ifndef gsscred_h
#define gsscred_h

typedef enum {
    IAKERB_NOT_CHECKED = 0,
    IAKERB_ACCESS_DENIED = 1,
    IAKERB_ACCESS_GRANTED = 2
} iakerb_access_status;

struct peer {
    xpc_connection_t peer;
    CFStringRef bundleID;
    CFStringRef callingAppBundleID;
    audit_token_t auditToken;  //the audit token of the calling app or app being impersonated
    struct HeimSession *session;
    bool needsManagedAppCheck;
    bool isManagedApp;
    CFStringRef currentDSID;
    iakerb_access_status access_status;
};

@protocol ManagedAppProvider <NSObject>

- (BOOL)isManagedApp:(NSString*)bundleId auditToken:(audit_token_t)auditToken;

@end

typedef NSString * (*HeimCredCurrentAltDSID)(void);
typedef bool (*HeimCredHasEntitlement)(struct peer *, const char *);
typedef uid_t (*HeimCredGetUid)(xpc_connection_t);
typedef NSData * (*HeimCredEncryptData)(NSData *);
typedef NSData * (*HeimCredDecryptData)(NSData *);
typedef au_asid_t (*HeimCredGetAsid)(xpc_connection_t);
typedef bool (*HeimCredVerifyAppleSigned)(struct peer *, NSString *);
typedef bool (*HeimCredSessionExists)(pid_t asid);
typedef void (*HeimCredSaveToDiskIfNeeded)(void);
typedef CFPropertyListRef (*HeimCredGetValueFromPreferences)(CFStringRef);
typedef void (*HeimExecuteOnRunQueue)(dispatch_block_t);

typedef struct {
    bool isMultiUser;
    HeimCredCurrentAltDSID currentAltDSID;
    HeimCredHasEntitlement hasEntitlement;
    HeimCredGetUid getUid;
    HeimCredGetAsid getAsid;
    HeimCredEncryptData encryptData;
    HeimCredDecryptData decryptData;
    HeimCredVerifyAppleSigned verifyAppleSigned;
    HeimCredSessionExists sessionExists;
    id<ManagedAppProvider> managedAppManager;
    bool useUidMatching;
    bool disableNTLMReflectionDetection;
    HeimCredSaveToDiskIfNeeded saveToDiskIfNeeded;
    HeimCredGetValueFromPreferences getValueFromPreferences;
    heim_ipc_event_callback_t expireFunction;
    heim_ipc_event_callback_t renewFunction;
    heim_ipc_event_final_t finalFunction;
    HeimCredNotifyCaches notifyCaches;
    time_t renewInterval;
    Class<GSSCredHelperClient> gssCredHelperClientClass;
    HeimExecuteOnRunQueue executeOnRunQueue;
} HeimCredGlobalContext;

extern HeimCredGlobalContext HeimCredGlobalCTX;

typedef CFDictionaryRef (*HeimCredAuthCallback)(struct peer *, HeimCredRef, CFDictionaryRef);

/*
 *
 */
struct HeimSession {
    CFRuntimeBase runtime;
    uid_t session;
    CFMutableDictionaryRef items;
    CFMutableDictionaryRef challenges;
    CFMutableDictionaryRef defaultCredentials;
    int updateDefaultCredential;
};

/*
 *
 */
struct HeimMech {
    CFRuntimeBase runtime;
    CFStringRef name;
    HeimCredStatusCallback statusCallback;
    HeimCredAuthCallback authCallback;
    HeimCredNotifyCaches notifyCaches;
    HeimCredTraceCallback traceCallback;
    bool readRestricted;
    CFArrayRef readOnlyCommands;
};

typedef enum {
    READ_SUCCESS = 0,
    READ_EMPTY = 1,
    READ_SIZE_ERROR = 2,
    READ_EXCEPTION = 3
} cache_read_status;

cache_read_status readCredCache(void);
void storeCredCache(void);
void notifyChangedCaches(void);

bool hasRenewTillInAttributes(CFDictionaryRef attributes);

void _HeimCredRegisterGeneric(void);

void _HeimCredRegisterConfiguration(void);
CFTypeRef GetValidatedValue(CFDictionaryRef object, CFStringRef key, CFTypeID requiredTypeID, CFErrorRef *error);

struct HeimSession * HeimCredCopySession(int sessionID);
void RemoveSession(au_asid_t asid);

void peer_final(void *ptr);

extern NSString *archivePath;

void do_Delete(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_SetAttrs(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_Auth(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_Scram(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_AddChallenge(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_CheckChallenge(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_Fetch(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_Query(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_GetDefault(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_Move(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_Status(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_DeleteAll(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_CreateCred(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_RetainCache(struct peer *peer, xpc_object_t request, xpc_object_t reply);
void do_ReleaseCache(struct peer *peer, xpc_object_t request, xpc_object_t reply);

bool isChallengeExpired(HeimCredRef cred);
bool checkNTLMChallenge(struct peer *peer, uint8_t challenge[8]);

CFTypeRef KerberosStatusCallback(HeimCredRef cred) CF_RETURNS_RETAINED;
CFTypeRef KerberosAcquireCredStatusCallback(HeimCredRef cred) CF_RETURNS_RETAINED;
CFTypeRef ConfigurationStatusCallback(HeimCredRef cred) CF_RETURNS_RETAINED;
CFDictionaryRef DefaultTraceCallback(CFDictionaryRef attributes) CF_RETURNS_RETAINED;

void
_HeimCredRegisterMech(CFStringRef mech,
		      CFSetRef publicAttributes,
		      HeimCredStatusCallback statusCallback,
		      HeimCredAuthCallback authCallback,
		      HeimCredNotifyCaches notifyCaches,
		      HeimCredTraceCallback traceCallback,
		      bool readRestricted,
		      CFArrayRef readOnlyCommands);

#endif /* gsscred_h */
