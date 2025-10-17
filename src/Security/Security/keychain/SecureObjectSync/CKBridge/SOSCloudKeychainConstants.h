/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
    These constants are used by the XPC service and its clients.
*/

#ifndef	_CKDXPC_CONSTANTS_H_
#define _CKDXPC_CONSTANTS_H_

__BEGIN_DECLS

extern const char *kCKPServiceName;

extern const char *kMessageKeyOperation;
extern const char *kMessageKeyKey;
extern const char *kMessageKeyValue;
extern const char *kMessageKeyError;
extern const char *kMessageKeyVersion;
extern const char *kMessageKeyGetNewKeysOnly;
extern const char *kMessageKeyKeysToGet;
extern const char *kMessageKeyKeysRequireFirstUnlock;
extern const char *kMessageKeyKeysRequiresUnlocked;
extern const char *kMessageOperationItemChanged;
extern const char *kMessageKeyNotificationFlags;
extern const char *kMessageKeyPeerIDList;
extern const char *kMessageKeyPeerID;
extern const char *kMesssgeKeyBackupPeerIDList;
extern const char *kMessageKeyAccountUUID;

extern const char *kMessageContext;
extern const char *kMessageKeyParameter;
extern const char *kMessageCircle;
extern const char *kMessageMessage;

extern const char *kMessageAlwaysKeys;
extern const char *kMessageFirstUnlocked;
extern const char *kMessageUnlocked;
extern const char *kMessageAllKeys;


extern const char *kOperationClearStore;
extern const char *kOperationSynchronize;
extern const char *kOperationSynchronizeAndWait;
extern const char *kOperationPUTDictionary;
extern const char *kOperationGETv2;
extern const char *kOperationRegisterKeys;
extern const char *kOperationRemoveKeys;

extern const char *kOperationHasPendingKey;

extern const uint64_t kCKDXPCVersion;

extern const char *kOperationFlush;

extern const char *kOperationPerfCounters;

extern const char *kOperationRequestSyncWithPeers;
extern const char *kOperationHasPendingSyncWithPeer;

extern const char *kOperationRequestEnsurePeerRegistration;

extern const char *kOperationGetPendingMesages;


extern const char * const kCloudKeychainStorechangeChangeNotification;

extern const char *kNotifyTokenForceUpdate;

#define kWAIT2MINID "EFRESH"

__END_DECLS

#endif	/* _CKDXPC_CONSTANTS_H_ */

