/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
/*
    This XPC service is essentially just a proxy to iCloud KVS, which exists since
    the main security code cannot link against Foundation.
    
    See sendTSARequestWithXPC in tsaSupport.c for how to call the service
    
    The client of an XPC service does not get connection events, nor does it
    need to deal with transactions.
*/

//------------------------------------------------------------------------------------------------


#include <CoreFoundation/CoreFoundation.h>
#include "SOSCloudKeychainConstants.h"

const uint64_t kCKDXPCVersion = 1;

// seems like launchd looks for the BundleIdentifier, not the name
const char *kCKPServiceName = "com.apple.security.cloudkeychainproxy3";   //"CloudKeychainProxy";

const char *kMessageKeyOperation = "operation";
const char *kMessageKeyKey = "key";
const char *kMessageKeyValue = "value";
const char *kMessageKeyError = "error";
const char *kMessageKeyVersion = "version";
const char *kMessageKeyGetNewKeysOnly = "GetNewKeysOnly";
const char *kMessageKeyKeysToGet = "KeysToGet";
const char *kMessageKeyKeysRequireFirstUnlock = "KeysRequireFirstUnlock";
const char *kMessageKeyKeysRequiresUnlocked = "KeysRequiresUnlocked";
const char *kMessageKeyNotificationFlags = "NotificationFlags";
const char *kMessageKeyPeerIDList = "peerIDList";
const char *kMesssgeKeyBackupPeerIDList = "backupPeerIDList";

/* parameters within the dictionary */
const char *kMessageAlwaysKeys = "AlwaysKeys";
const char *kMessageFirstUnlocked = "FirstUnlockKeys";
const char *kMessageUnlocked = "UnlockedKeys";

const char *kMessageContext = "Context";
const char *kMessageAllKeys = "AllKeys";
const char *kMessageKeyParameter = "KeyParameter";
const char *kMessageCircle = "Circle";
const char *kMessageMessage = "Message";
const char *kMessageKeyDeviceName = "deviceName";
const char *kMessageKeyPeerID = "peerID";
const char *kMessageKeySendersPeerID = "sendersPeerID";
const char *kMessageKeyAccountUUID = "AcctUUID";

const char *kMessageOperationItemChanged = "ItemChanged";

const char *kOperationClearStore = "ClearStore";
const char *kOperationSynchronize = "Synchronize";
const char *kOperationSynchronizeAndWait = "SynchronizeAndWait";

const char *kOperationFlush = "Flush";

const char *kOperationPerfCounters = "PerfCounters";

const char *kOperationPUTDictionary = "PUTDictionary";
const char *kOperationGETv2 = "GETv2";

const char *kOperationRegisterKeys = "RegisterKeys";
const char *kOperationRemoveKeys = "RemoveKeys";

const char *kOperationHasPendingKey = "hasPendingKey";

const char *kOperationRequestSyncWithPeers = "requestSyncWithPeers";
const char *kOperationHasPendingSyncWithPeer = "hasPendingSyncWithPeer";
const char *kOperationRequestEnsurePeerRegistration = "requestEnsurePeerRegistration";


/*
    The values for the KVS notification and KVS Store ID must be identical to the values
    in syncdefaultsd (SYDApplication.m). The notification string is used in two places:
    it is in our launchd plist (com.apple.security.cloudkeychainproxy.plist) as the
    LaunchEvents/com.apple.notifyd.matching key and is examined in code in the stream event handler.

    The KVS Store ID (_SYDRemotePreferencesStoreIdentifierKey in SYDApplication.m) must
    be in the entitlements. The bundle identifier is (com.apple.security.cloudkeychainproxy3)
    is used by installInfoForBundleIdentifiers in SYDApplication.m and is used to look up our
    daemon to figure out what store to use, etc.
*/

const char * const kCloudKeychainStorechangeChangeNotification = "com.apple.security.cloudkeychainproxy.kvstorechange3"; // was "com.apple.security.cloudkeychain.kvstorechange" for seeds

const char *kNotifyTokenForceUpdate = "com.apple.security.cloudkeychain.forceupdate";
