/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#include <TargetConditionals.h>

#if TARGET_OS_OSX
#include <CoreFoundation/CoreFoundation.h>

#include <sys/types.h>
#include <pwd.h>
#include <uuid/uuid.h>
#include "iOSforOSX.h"
#include <pwd.h>
#include <unistd.h>

// Was in SOSAccount.c
#define SEC_CONST_DECL(k,v) const CFStringRef k = CFSTR(v);
// We may not have all of these we need
SEC_CONST_DECL (kSecAttrAccessible, "pdmn");
SEC_CONST_DECL (kSecAttrAccessibleAlwaysThisDeviceOnly, "dku");
SEC_CONST_DECL (kSecAttrAccessibleAlwaysThisDeviceOnlyPrivate, "dku");
SEC_CONST_DECL (kSecAttrAccessControl, "accc");
SEC_CONST_DECL (kSecAttrTokenID, "tkid");
SEC_CONST_DECL (kSecAttrAccessGroupToken, "com.apple.token");
SEC_CONST_DECL (kSecUseCredentialReference, "u_CredRef");
SEC_CONST_DECL (kSecUseOperationPrompt, "u_OpPrompt");
SEC_CONST_DECL (kSecUseNoAuthenticationUI, "u_NoAuthUI");
SEC_CONST_DECL (kSecUseAuthenticationUI, "u_AuthUI");
SEC_CONST_DECL (kSecUseAuthenticationUIAllow, "u_AuthUIA");
SEC_CONST_DECL (kSecUseAuthenticationUIFail, "u_AuthUIF");
SEC_CONST_DECL (kSecUseAuthenticationUISkip, "u_AuthUIS");
SEC_CONST_DECL (kSecUseAuthenticationContext, "u_AuthCtx");
SEC_CONST_DECL (kSecUseToken, "u_Token");
SEC_CONST_DECL (kSecUseCallerName, "u_CallerName");

#endif
