/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
 * SOSCloudKeychainClient.h -  Implementation of the transport layer from CKBridge to SOSAccount/SOSCircle
 */

/*!
    @header SOSCloudKeychainClient
    The functions provided in SOSCloudTransport.h provide an interface
    from CKBridge to SOSAccount/SOSCircle
 */

#ifndef _SOSCLOUDKEYCHAINCLIENT_H_
#define _SOSCLOUDKEYCHAINCLIENT_H_

#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFError.h>
#include <dispatch/dispatch.h>

__BEGIN_DECLS


// MARK: ---------- SOSCloudTransport ----------

enum
{
    kSOSObjectNotFoundError = 1,
    kSOSObjectCantBeConvertedToXPCObject,
    kSOSOUnexpectedConnectionEvent,
    kSOSOXPCErrorEvent,
    kSOSOUnexpectedXPCEvent,
    kSOSConnectionNotOpen,
    kSOSNoServerReply,
    kSOSMessageInvalid,
    kSOSMessageNotSupported,
};

typedef CFArrayRef (^CloudItemsChangedBlock)(CFDictionaryRef values);
typedef void (^CloudKeychainReplyBlock)(CFDictionaryRef returnedValues, CFErrorRef error);

/* SOSCloudTransport protocol.  */
typedef struct SOSCloudTransport *SOSCloudTransportRef;
struct SOSCloudTransport
{
    void (*put)(SOSCloudTransportRef transport, CFDictionaryRef valuesToPut, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    void (*updateKeys)(SOSCloudTransportRef transport, CFDictionaryRef keys, CFStringRef accountUUID, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

    void (*getDeviceID)(SOSCloudTransportRef transport, CloudKeychainReplyBlock replyBlock);

    // Debug calls
    void (*get)(SOSCloudTransportRef transport, CFArrayRef keysToGet, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    void (*getAll)(SOSCloudTransportRef transport, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    void (*synchronize)(SOSCloudTransportRef transport, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    void (*synchronizeAndWait)(SOSCloudTransportRef transport, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

    void (*clearAll)(SOSCloudTransportRef transport, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    void (*removeObjectForKey)(SOSCloudTransportRef transport, CFStringRef keyToRemove, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

    bool (*hasPendingKey)(SOSCloudTransportRef transport, CFStringRef keyName, CFErrorRef* error);

    void (*requestSyncWithPeers)(SOSCloudTransportRef transport, CFArrayRef /* CFStringRef */ peers, CFArrayRef /* CFStringRef */ backupPeers, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

    void (*requestEnsurePeerRegistration)(SOSCloudTransportRef transport, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    void (*requestPerfCounters)(SOSCloudTransportRef transport, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    void (*flush)(SOSCloudTransportRef transport, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    const void *itemsChangedBlock;
    void (*removeKeys)(SOSCloudTransportRef transport, CFArrayRef keys, CFStringRef accountUUID, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
    void (*counters)(SOSCloudTransportRef transport, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

};

void SOSCloudKeychainUpdateKeys(CFDictionaryRef keys, CFStringRef accountUUID, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

void SOSCloudKeychainPutObjectsInCloud(CFDictionaryRef objects, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

void SOSCloudKeychainSetItemsChangedBlock(CloudItemsChangedBlock itemsChangedBlock);

CF_RETURNS_RETAINED CFArrayRef SOSCloudKeychainHandleUpdateMessage(CFDictionaryRef updates);
void SOSCloudKeychainSynchronizeAndWait(dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

// Debug only?

void SOSCloudKeychainGetObjectsFromCloud(CFArrayRef keysToGet, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
void SOSCloudKeychainSynchronize(dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
void SOSCloudKeychainClearAll(dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
void SOSCloudKeychainGetAllObjectsFromCloud(dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
bool SOSCloudKeychainHasPendingKey(CFStringRef keyName, CFErrorRef* error);

void SOSCloudKeychainRequestSyncWithPeers(CFArrayRef /* CFStringRef */ peers, CFArrayRef /* CFStringRef */ backupPeers, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

void SOSCloudKeychainRequestEnsurePeerRegistration(dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
void SOSCloudKeychainRequestPerfCounters(dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

void SOSCloudKeychainFlush(dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);
CFDictionaryRef SOSCloudCopyKVSState(void);
void SOSCloudKeychainRemoveKeys(CFArrayRef keys, CFStringRef accountUUID, dispatch_queue_t processQueue, CloudKeychainReplyBlock replyBlock);

// For unit tests
void
SOSCloudTransportSetDefaultTransport(SOSCloudTransportRef transport);

__END_DECLS

#endif /* !_SOSCLOUDKEYCHAINCLIENT_H_ */

