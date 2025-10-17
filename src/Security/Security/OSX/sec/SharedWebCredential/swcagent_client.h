/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
#ifndef	_SWCAGENT_CLIENT_H_
#define _SWCAGENT_CLIENT_H_

#include <stdint.h>

#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFError.h>

#include <xpc/xpc.h>
#include <CoreFoundation/CFXPCBridge.h>

// TODO: This should be in client of XPC code locations...
#define kSWCAXPCServiceName "com.apple.security.swcagent"

//
// MARK: XPC Information.
//

extern CFStringRef sSWCAXPCErrorDomain;

//
// MARK: XPC Interfaces
//

extern const char *kSecXPCKeyOperation;
extern const char *kSecXPCKeyResult;
extern const char *kSecXPCKeyEndpoint;
extern const char *kSecXPCKeyError;
extern const char *kSecXPCKeyClientToken;
extern const char *kSecXPCKeyPeerInfoArray;
extern const char *kSecXPCKeyPeerInfo;
extern const char *kSecXPCKeyUserLabel;
extern const char *kSecXPCKeyDSID;
extern const char *kSecXPCKeyUserPassword;
extern const char *kSecXPCLimitInMinutes;
extern const char *kSecXPCKeyQuery;
extern const char *kSecXPCKeyAttributesToUpdate;
extern const char *kSecXPCKeyDomain;
extern const char *kSecXPCKeyDigest;
extern const char *kSecXPCKeyCertificate;
extern const char *kSecXPCKeySettings;
extern const char *kSecXPCKeyEscrowLabel;
extern const char *kSecXPCKeyTriesLabel;
extern const char *kSecXPCKeyViewName;
extern const char *kSecXPCKeyViewActionCode;

//
// MARK: Mach port request IDs
//
enum SWCAXPCOperation {
    swca_add_request_id,
    swca_update_request_id,
    swca_delete_request_id,
    swca_copy_request_id,
    swca_select_request_id, // deprecated: no longer used
    swca_copy_pairs_request_id,
    swca_set_selection_request_id,
    swca_enabled_request_id,
};

xpc_object_t swca_message_with_reply_sync(xpc_object_t message, CFErrorRef *error);
xpc_object_t swca_create_message(enum SWCAXPCOperation op, CFErrorRef *error);
bool swca_message_no_error(xpc_object_t message, CFErrorRef *error);
long swca_message_response(xpc_object_t replyMessage, CFErrorRef *error);

bool swca_autofill_enabled(const audit_token_t *auditToken);

bool swca_confirm_operation(enum SWCAXPCOperation op,
                            const audit_token_t *auditToken,
                            CFTypeRef query,
                            CFErrorRef *error,
                            void (^add_negative_entry)(CFStringRef fqdn));

CFTypeRef swca_message_copy_response(xpc_object_t replyMessage, CFErrorRef *error);

CFDictionaryRef swca_copy_selected_dictionary(enum SWCAXPCOperation op,
                                              const audit_token_t *auditToken,
                                              CFTypeRef items,
                                              CFErrorRef *error);

CFArrayRef swca_copy_pairs(enum SWCAXPCOperation op,
                           const audit_token_t *auditToken,
                           CFErrorRef *error);

bool swca_set_selection(enum SWCAXPCOperation op,
                        const audit_token_t *auditToken,
                        CFTypeRef dictionary,
                        CFErrorRef *error);

bool swca_send_sync_and_do(enum SWCAXPCOperation op, CFErrorRef *error,
                                bool (^add_to_message)(xpc_object_t message, CFErrorRef* error),
                                bool (^handle_response)(xpc_object_t response, CFErrorRef* error));


#endif /* _SWCAGENT_CLIENT_H_ */
