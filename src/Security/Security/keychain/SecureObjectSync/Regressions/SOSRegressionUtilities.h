/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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
//  SOSRegressionUtilities.h
//

#ifndef sec_SOSRegressionUtilities_h
#define sec_SOSRegressionUtilities_h

#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFError.h>
#include <Security/SecKey.h>
#include <Security/SecureObjectSync/SOSPeerInfo.h>
#include "keychain/SecureObjectSync/SOSPeerInfoPriv.h"
#include "keychain/SecureObjectSync/SOSFullPeerInfo.h"
#include "keychain/SecureObjectSync/SOSCircle.h"
#include "keychain/SecureObjectSync/SOSCirclePriv.h"
#include <TargetConditionals.h>

__BEGIN_DECLS

#define SOS_ENABLED (!TARGET_OS_XR && (TARGET_OS_OSX || TARGET_OS_IOS))

CFStringRef myMacAddress(void);
const char *cfabsoluteTimeToString(CFAbsoluteTime abstime);
const char *cfabsoluteTimeToStringLocal(CFAbsoluteTime abstime);
bool XPCServiceInstalled(void);

void registerForKVSNotifications(const void *observer, CFStringRef name, CFNotificationCallback callBack);
void unregisterFromKVSNotifications(const void *observer);

bool testPutObjectInCloudAndSync(CFStringRef key, CFTypeRef object, CFErrorRef *error, dispatch_group_t dgroup, dispatch_queue_t processQueue);
bool testPutObjectInCloud(CFStringRef key, CFTypeRef object, CFErrorRef *error, dispatch_group_t dgroup, dispatch_queue_t processQueue);

CFTypeRef testGetObjectFromCloud(CFStringRef key, dispatch_queue_t processQueue, dispatch_group_t dgroup);
CFTypeRef testGetObjectsFromCloud(CFArrayRef keys, dispatch_queue_t processQueue, dispatch_group_t dgroup);

bool testSynchronize(dispatch_queue_t processQueue, dispatch_group_t dgroup);
bool testClearAll(dispatch_queue_t processQueue, dispatch_group_t dgroup);

//
// MARK: Peer Info helpers
//   These generate keys for your and create info objects with that name.
//

CFDictionaryRef SOSCreatePeerGestaltFromName(CFStringRef name);

SOSPeerInfoRef
SOSCreatePeerInfoFromName(CFStringRef name,
                          SecKeyRef* outSigningKey,
                          SecKeyRef* outOctagonSigningKey,
                          SecKeyRef* outOctagonEncryptionKey,
                          CFErrorRef *error);

SOSFullPeerInfoRef
SOSCreateFullPeerInfoFromName(CFStringRef name,
                              SecKeyRef* outSigningKey,
                              SecKeyRef* outOctagonSigningKey,
                              SecKeyRef* outOctagonEncryptionKey,
                              CFErrorRef *error);

SOSFullPeerInfoRef SOSTestV0FullPeerInfo(CFStringRef name, SecKeyRef userKey, CFStringRef OSName, SOSPeerInfoDeviceClass devclass);
SOSFullPeerInfoRef SOSTestFullPeerInfo(CFStringRef name, SecKeyRef userKey, CFStringRef OSName, SOSPeerInfoDeviceClass devclass);
SOSCircleRef SOSTestCircle(SecKeyRef userKey, void * firstFpiv, ... );
SecKeyRef SOSMakeUserKeyForPassword(const char *passwd);
bool SOSPeerValidityCheck(SOSFullPeerInfoRef fpi, SecKeyRef userKey, CFErrorRef *error);


CFStringRef SOSModelFromType(SOSPeerInfoDeviceClass cls);

__END_DECLS

#endif /* sec_SOSRegressionUtilities_h */

