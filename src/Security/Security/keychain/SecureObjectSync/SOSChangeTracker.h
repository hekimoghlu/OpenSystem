/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
/*!
 @header SOSChangeTracker.h - Manifest caching and change propagation.
 */

#ifndef _SEC_SOSCHANGETRACKER_H_
#define _SEC_SOSCHANGETRACKER_H_

#include "keychain/SecureObjectSync/SOSDataSource.h"

#include "utilities/simulatecrash_assert.h"

__BEGIN_DECLS

enum {
    kSOSErrorNotConcreteError = 1042,
};


//
// Interface to encoding and decoding changes
//

typedef CFTypeRef SOSChangeRef;

static inline void SOSChangesAppendDelete(CFMutableArrayRef changes, CFTypeRef object) {
    const void *values[] = { object };
    SOSChangeRef change = CFArrayCreate(kCFAllocatorDefault, values, array_size(values), &kCFTypeArrayCallBacks);
    CFArrayAppendValue(changes, change);
    CFReleaseSafe(change);
}

static inline void SOSChangesAppendAdd(CFMutableArrayRef changes, CFTypeRef object) {
    CFArrayAppendValue(changes, object);
}

// Return the object and return true if it's an add and false if it's a delete.
static inline bool SOSChangeGetObject(SOSChangeRef change, CFTypeRef *object) {
    if (CFGetTypeID(change) == CFArrayGetTypeID()) {
        assert(CFArrayGetCount(change) == 1);
        *object = CFArrayGetValueAtIndex(change, 0);
        return false;
    } else {
        *object = change;
        return true;
    }
}

CFDataRef SOSChangeCopyDigest(SOSDataSourceRef dataSource, SOSChangeRef change, bool *isDel, SOSObjectRef *object, CFErrorRef *error);

CFStringRef SOSChangeCopyDescription(SOSChangeRef change);

CFStringRef SOSChangesCopyDescription(CFArrayRef changes);

//
// ChangeTracker
//

typedef struct __OpaqueSOSChangeTracker *SOSChangeTrackerRef;

SOSChangeTrackerRef SOSChangeTrackerCreate(CFAllocatorRef allocator, bool isConcrete, CFArrayRef children, CFErrorRef *error);

// Change the concreteness of the current ct (a non concrete ct does not support SOSChangeTrackerCopyManifest().
void SOSChangeTrackerSetConcrete(SOSChangeTrackerRef ct, bool isConcrete);

typedef bool(^SOSChangeTrackerUpdatesChanges)(SOSChangeTrackerRef ct, SOSEngineRef engine, SOSTransactionRef txn, SOSDataSourceTransactionSource source, SOSDataSourceTransactionPhase phase, CFArrayRef changes, CFErrorRef *error);

void SOSChangeTrackerRegisterChangeUpdate(SOSChangeTrackerRef ct, SOSChangeTrackerUpdatesChanges child);

typedef bool(^SOSChangeTrackerUpdatesManifests)(SOSChangeTrackerRef ct, SOSEngineRef engine, SOSTransactionRef txn, SOSDataSourceTransactionSource source, SOSDataSourceTransactionPhase phase, SOSManifestRef removals, SOSManifestRef additions, CFErrorRef *error);

void SOSChangeTrackerRegisterManifestUpdate(SOSChangeTrackerRef ct, SOSChangeTrackerUpdatesManifests child);

// Remove any blocks registered though either SOSChangeTrackerRegisterChangeUpdate or SOSChangeTrackerRegisterManifestUpdate
void SOSChangeTrackerResetRegistration(SOSChangeTrackerRef ct);

// Set the manifest for this changeTracker, causing it to be updated whenever changes pass by.
// Set the manifest to NULL to stop tracking changes (but not forwarding to children).
void SOSChangeTrackerSetManifest(SOSChangeTrackerRef ct, SOSManifestRef manifest);

// Return a snapshot of the current manifest of this ct.
SOSManifestRef SOSChangeTrackerCopyManifest(SOSChangeTrackerRef ct, CFErrorRef *error);

// Apply changes to the (cached) manifest, and notify all children accordingly
bool SOSChangeTrackerTrackChanges(SOSChangeTrackerRef ct, SOSEngineRef engine, SOSTransactionRef txn, SOSDataSourceTransactionSource source, SOSDataSourceTransactionPhase phase, CFArrayRef changes, CFErrorRef *error);

__END_DECLS

#endif /* !_SEC_SOSCHANGETRACKER_H_ */
