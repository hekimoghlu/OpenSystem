/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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

//
//  SecLogSettingsServer.c
//  sec
//
//

#include "keychain/securityd/SecLogSettingsServer.h"
#include "keychain/SecureObjectSync/SOSAccountPriv.h"
#include <Security/SecBase.h>
#include <Security/SecLogging.h>
#include "keychain/SecureObjectSync/SOSTransportCircle.h"
#include <utilities/debugging.h>
#include <utilities/SecCFWrappers.h>
#include <utilities/SecCFError.h>

CFPropertyListRef
SecCopyLogSettings_Server(CFErrorRef* error)
{
    return CopyCurrentScopePlist();
}

bool
SecSetXPCLogSettings_Server(CFTypeRef type, CFErrorRef* error)
{
    bool success = false;
    if (isString(type)) {
        ApplyScopeListForID(type, kScopeIDXPC);
        success = true;
    } else if (isDictionary(type)) {
        ApplyScopeDictionaryForID(type, kScopeIDXPC);
        success = true;
    } else {
        success = SecError(errSecParam, error, CFSTR("Unsupported CFType"));
    }

    return success;
}

bool
SecSetCircleLogSettings_Server(CFTypeRef type, CFErrorRef* error)
{
    bool success = false;
    SOSAccount* account = (__bridge SOSAccount*)SOSKeychainAccountGetSharedAccount();
    if (account) {
        success = SOSAccountPostDebugScope(account, type, error);
    }
    return success;
}

