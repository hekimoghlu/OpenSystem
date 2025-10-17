/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#include "keychain/securityd/SecItemBackupServer.h"
#include "keychain/securityd/SecItemServer.h"
#include "keychain/SecureObjectSync/SOSEnginePriv.h"
#include "keychain/SecureObjectSync/SOSPeer.h"
#include <Security/SecureObjectSync/SOSBackupSliceKeyBag.h>
#include <Security/SecureObjectSync/SOSViews.h>
#include <unistd.h>

#include "keychain/securityd/SecDbItem.h"
#include <utilities/der_plist.h>

static bool withDataSourceAndEngine(CFErrorRef *error, void (^action)(SOSDataSourceRef ds, SOSEngineRef engine)) {
    bool ok = false;
    SOSDataSourceFactoryRef dsf = SecItemDataSourceFactoryGetDefault();
    SOSDataSourceRef ds = SOSDataSourceFactoryCreateDataSource(dsf, kSecAttrAccessibleWhenUnlocked, error);
    if (ds) {
        SOSEngineRef engine = SOSDataSourceGetSharedEngine(ds, error);
        if (engine) {
            action(ds, engine);
            ok = true;
        }
        ok &= SOSDataSourceRelease(ds, error);
    }
    return ok;
}

int SecServerItemBackupHandoffFD(CFStringRef backupName, CFErrorRef *error) {
    __block int fd = -1;
    if (!withDataSourceAndEngine(error, ^(SOSDataSourceRef ds, SOSEngineRef engine) {
        SOSEngineForPeerID(engine, backupName, error, ^(SOSTransactionRef txn, SOSPeerRef peer) {
            fd = SOSPeerHandoffFD(peer, error);
        });
    }) && fd >= 0) {
        close(fd);
        fd = -1;
    }
    return fd;
}

bool SecServerItemBackupSetConfirmedManifest(CFStringRef backupName, CFDataRef keybagDigest, CFDataRef manifestData, CFErrorRef *error) {
    __block bool ok = true;
    ok &= withDataSourceAndEngine(error, ^(SOSDataSourceRef ds, SOSEngineRef engine) {
        ok = SOSEngineSetPeerConfirmedManifest(engine, backupName, keybagDigest, manifestData, error);
    });
    return ok;
}

CFArrayRef SecServerItemBackupCopyNames(CFErrorRef *error) {
    __block CFArrayRef names = NULL;
    if (!withDataSourceAndEngine(error, ^(SOSDataSourceRef ds, SOSEngineRef engine) {
        names = SOSEngineCopyBackupPeerNames(engine, error);
    })) {
        CFReleaseNull(names);
    }
    return names;
}

CFStringRef SecServerItemBackupEnsureCopyView(CFStringRef viewName, CFErrorRef *error) {
    __block CFStringRef name = NULL;
    if(!withDataSourceAndEngine(error, ^(SOSDataSourceRef ds, SOSEngineRef engine) {
        name = SOSEngineEnsureCopyBackupPeerForView(engine, viewName, error);
    })) {
        CFReleaseNull(name);
    }
    return name;
}

// TODO Move to datasource and remove dsRestoreObject
static bool SOSDataSourceWithBackup(SOSDataSourceRef ds, CFDataRef backup, keybag_handle_t bag_handle, CFErrorRef *error, void(^with)(SOSObjectRef item)) {
    __block bool ok = true;
    CFPropertyListRef plist = CFPropertyListCreateWithDERData(kCFAllocatorDefault, backup, kCFPropertyListImmutable, NULL, error);
    CFDictionaryRef bdict = asDictionary(plist, error);
    ok = (bdict != NULL);
    if (ok) CFDictionaryForEach(bdict, ^(const void *key, const void *value) {
        CFStringRef className = asString(key, error);
        if (className) {
            const SecDbClass *cls = kc_class_with_name(className);
            if (cls) {
                CFArrayRef items = asArray(value, error);
                CFDataRef edata;
                if (items) CFArrayForEachC(items, edata) {
                    SOSObjectRef item = (SOSObjectRef)SecDbItemCreateWithEncryptedData(kCFAllocatorDefault, cls, edata, bag_handle, NULL, error);
                    if (item) {
                        with(item);
                        CFRelease(item);
                    } else {
                        ok = false;
                    }
                } else {
                    ok = false;
                }
            } else {
                ok &= SecError(errSecDecode, error, CFSTR("bad class %@ in backup"), className);
            }
        } else {
            ok = false;
        }
    });
    CFReleaseSafe(plist);
    return ok;
}

bool SecServerItemBackupRestore(CFStringRef backupName, CFStringRef peerID, CFDataRef keybag, CFDataRef secret, CFDataRef backup, CFErrorRef *error) {
    // TODO: Decrypt and merge items in backup to dataSource

    __block bool ok = false; // return false if the bag_handle code fails.
    CFDataRef aksKeybag = NULL;
    CFMutableSetRef viewSet = NULL;
    SOSBackupSliceKeyBagRef backupSliceKeyBag = NULL;
    keybag_handle_t bag_handle = bad_keybag_handle;

    require(asData(secret, error), xit);
    require(backupSliceKeyBag = SOSBackupSliceKeyBagCreateFromData(kCFAllocatorDefault, keybag, error), xit);

    if (peerID) {
        bag_handle = SOSBSKBLoadAndUnlockWithPeerIDAndSecret(backupSliceKeyBag, peerID, secret, error);
    } else {
        if (SOSBSKBIsDirect(backupSliceKeyBag)) {
            bag_handle = SOSBSKBLoadAndUnlockWithDirectSecret(backupSliceKeyBag, secret, error);
        } else {
            bag_handle = SOSBSKBLoadAndUnlockWithWrappingSecret(backupSliceKeyBag, secret, error);
        }
    }
    require(bag_handle != bad_keybag_handle, xit);

    // TODO: How do we know which views we are allow to restore
    //viewSet = SOSAccountCopyRestorableViews();

    ok = true; // Start from original code start point - otherwise begin in this nest of stuff
    ok &= withDataSourceAndEngine(error, ^(SOSDataSourceRef ds, SOSEngineRef engine) {
        ok &= SOSDataSourceWith(ds, error, ^(SOSTransactionRef txn, bool *commit) {
            ok &= SOSDataSourceWithBackup(ds, backup, bag_handle, error, ^(SOSObjectRef item) {
                //if (SOSDataSourceIsInViewSet(item, viewSet)) {
                    SOSObjectRef mergedItem = NULL;
                    if (SOSDataSourceMergeObject(ds, txn, item, &mergedItem, error)) {
                        // if mergedItem == item then it was restored otherwise it was rejected by the conflict resolver.
                        CFReleaseSafe(mergedItem);
                    }
                //}
            });
        });
    });

xit:
    if (bag_handle != bad_keybag_handle)
        ok &= ks_close_keybag(bag_handle, error);

    CFReleaseSafe(backupSliceKeyBag);
    CFReleaseSafe(aksKeybag);
    CFReleaseSafe(viewSet);

    return ok;
}




