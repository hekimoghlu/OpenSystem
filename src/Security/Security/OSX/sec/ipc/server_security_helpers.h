/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#ifndef server_security_helpers_h
#define server_security_helpers_h

#include <Security/SecTask.h>
#include "ipc/securityd_client.h"

typedef enum SecSecurityClientKeychainSharingState {
    SecSecurityClientKeychainSharingStateDisabled = 0,
    SecSecurityClientKeychainSharingStateEnabled = 1,
} SecSecurityClientKeychainSharingState;

CFTypeRef SecCreateLocalCFSecuritydXPCServer(void);
void SecAddLocalSecuritydXPCFakeEntitlement(CFStringRef entitlement, CFTypeRef value);
void SecResetLocalSecuritydXPCFakeEntitlements(void);
void SecCreateSecuritydXPCServer(void);

bool fill_security_client(SecurityClient * client, const uid_t uid, audit_token_t auditToken);
CFArrayRef SecTaskCopyAccessGroups(SecTaskRef task);

bool SecFillSecurityClientMuser(SecurityClient *client);

/*!
 @function SecTaskIsEligiblePlatformBinary
 @abstract Determine whether task belongs to valid platform binary and optionally has one of the allowed identifiers.
 @param task The client task to be evaluated.
 @param identifiers Optional array of codesigning identifiers of allowed callers. Pass NULL to permit any platform binary.
 @result Client satisfies the criteria or not.
 */
bool SecTaskIsEligiblePlatformBinary(SecTaskRef task, CFArrayRef identifiers);

// Testing support
void SecAccessGroupsSetCurrent(CFArrayRef accessGroups);
void SecSecurityClientRegularToAppClip(void);
void SecSecurityClientAppClipToRegular(void);
void SecSecurityClientSetApplicationIdentifier(CFStringRef identifier);
void SecSecurityClientSetKeychainSharingState(SecSecurityClientKeychainSharingState state);

#if TARGET_OS_IOS && HAVE_MOBILE_KEYBAG_SUPPORT
bool device_is_multiuser(void);
#endif

#endif /* server_security_helpers_h */
