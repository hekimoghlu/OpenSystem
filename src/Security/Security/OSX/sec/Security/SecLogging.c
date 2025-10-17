/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 2, 2023.
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
#include <Security/SecLogging.h>
#include "securityd_client.h"
#include "SecuritydXPC.h"
#include <os/activity.h>

static bool dict_to_error_request(enum SecXPCOperation op, CFDictionaryRef query, CFErrorRef *error)
{
    return securityd_send_sync_and_do(op, error, ^bool(xpc_object_t message, CFErrorRef *error) {
        return SecXPCDictionarySetPList(message, kSecXPCKeyQuery, query, error);
    }, NULL);
}

static CFDictionaryRef void_to_dict_error_request(enum SecXPCOperation op, CFErrorRef *error) {
    __block CFDictionaryRef dict = NULL;
    securityd_send_sync_and_do(op, error, NULL, ^bool(xpc_object_t response, CFErrorRef *error) {
        return (dict = SecXPCDictionaryCopyPList(response, kSecXPCKeyResult, error));
    });
    return dict;
}

CFArrayRef SecGetCurrentServerLoggingInfo(CFErrorRef *error)
{
    __block CFArrayRef result;
    os_activity_initiate("SecGetCurrentServerLoggingInfo", OS_ACTIVITY_FLAG_DEFAULT, ^{
        result = SECURITYD_XPC(sec_get_log_settings, void_to_dict_error_request, error);
    });
    return result;
}

bool SecSetLoggingInfoForXPCScope(CFPropertyListRef /* String or Dictionary of strings */ settings, CFErrorRef *error)
{
    __block bool result;
    os_activity_initiate("SecSetLoggingInfoForXPCScope", OS_ACTIVITY_FLAG_DEFAULT, ^{
        result = SECURITYD_XPC(sec_set_xpc_log_settings, dict_to_error_request, settings, error);
    });
    return result;
}

bool SecSetLoggingInfoForCircleScope(CFPropertyListRef /* String or Dictionary of strings */ settings, CFErrorRef *error)
{
    __block bool result;
    os_activity_initiate("SecSetLoggingInfoForCircleScope", OS_ACTIVITY_FLAG_DEFAULT, ^{
        result = SECURITYD_XPC(sec_set_circle_log_settings, dict_to_error_request, settings, error);
    });
    return result;
}
