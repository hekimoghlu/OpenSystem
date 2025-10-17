/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#ifndef _EAP8021X_SIMACCESS_PRIVATE_H
#define _EAP8021X_SIMACCESS_PRIVATE_H

#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFRunLoop.h>
#include "EAPSIMAKA.h"

#if TARGET_OS_IPHONE

CFStringRef
_SIMCopyIMSI(CFDictionaryRef properties);

CFStringRef
_SIMCopyRealm(CFDictionaryRef properties);

CFDictionaryRef
_SIMCopyEncryptedIMSIInfo(EAPType type);

Boolean
_SIMIsOOBPseudonymSupported(Boolean *isSupported);

CFStringRef
_SIMCopyOOBPseudonym(void);

void
_SIMReportDecryptionError(CFDataRef encryptedIdentity);

CFDictionaryRef
_SIMCreateAuthResponse(CFStringRef slotUUID, CFDictionaryRef auth_params);

typedef void (*SIMAccessConnectionCallback)(CFTypeRef connection, CFStringRef status, void* info);

CFTypeRef
_SIMAccessConnectionCreate(SIMAccessConnectionCallback callback, void *info);

#endif /* TARGET_OS_IPHONE */

#endif /* _EAP8021X_SIMACCESS_PRIVATE_H */
