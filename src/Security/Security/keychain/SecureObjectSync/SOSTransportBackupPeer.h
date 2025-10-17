/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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


#ifndef SOSTransportBackupPeer_h
#define SOSTransportBackupPeer_h

#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFRuntime.h>
#include "keychain/SecureObjectSync/SOSAccount.h"

typedef struct __OpaqueSOSTransportBackupPeer *SOSTransportBackupPeerRef;


struct __OpaqueSOSTransportBackupPeer {
    CFRuntimeBase   _base;
    CFStringRef     fileLocation;

};

CFIndex SOSTransportBackupPeerGetTransportType(SOSTransportBackupPeerRef transport, CFErrorRef *error);
SOSTransportBackupPeerRef SOSTransportBackupPeerCreate(CFStringRef fileLocation, CFErrorRef *error);

#endif
