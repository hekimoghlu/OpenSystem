/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
//  SOSEnginePriv.h
//  sec
//
//

#ifndef SOSEnginePriv_h
#define SOSEnginePriv_h

#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFRuntime.h>
#include "keychain/SecureObjectSync/SOSEngine.h"

/* SOSEngine implementation. */
struct __OpaqueSOSEngine {
    CFRuntimeBase _base;
    SOSDataSourceRef dataSource;
    CFStringRef myID;                       // My peerID in the circle
    // We need to address the issues of corrupt keychain items
    SOSManifestRef unreadable;              // Possibly by having a set of unreadable items, to which we
    // add any corrupted items in the db that have yet to be deleted.
    // This happens if we notce corruption during a (read only) query.
    // We would also perma-subtract unreadable from manifest whenever
    // anyone asked for manifest.  This result would be cached in
    // The manifestCache below, so we just need a key into the cache
    CFDataRef localMinusUnreadableDigest;   // or a digest (CFDataRef of the right size).

    CFMutableDictionaryRef manifestCache;       // digest -> ( refcount, manifest )
    CFMutableDictionaryRef peerMap;             // peerId -> SOSPeerRef
    CFDictionaryRef viewNameSet2ChangeTracker;  // CFSetRef of CFStringRef -> SOSChangeTrackerRef
    CFDictionaryRef viewName2ChangeTracker;     // CFStringRef -> SOSChangeTrackerRef
    CFArrayRef peerIDs;
    CFDateRef lastTraceDate;                    // Last time we did a CloudKeychainTrace
    CFMutableDictionaryRef coders;
    bool haveLoadedCoders;

    bool codersNeedSaving;
    dispatch_queue_t queue;                     // Engine queue

    dispatch_source_t save_timer;               // Engine state save timer
    dispatch_queue_t syncCompleteQueue;              // Non-retained queue for async notificaion
    SOSEnginePeerInSyncBlock syncCompleteListener;   // Block to call to notify the listener.

    bool save_timer_pending;                    // Engine state timer running, read/modify on engine queue

};

#endif /* SOSEnginePriv_h */
