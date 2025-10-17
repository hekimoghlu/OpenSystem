/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#ifndef _OTATRUSTUTILITIES_H_
#define _OTATRUSTUTILITIES_H_  1

#include <CoreFoundation/CoreFoundation.h>
#include <sys/types.h>
#include <stdio.h>
#include <dispatch/dispatch.h>

__BEGIN_DECLS

// Opaque type that holds the data for a specific version of the OTA PKI assets
typedef struct _OpaqueSecOTAPKI *SecOTAPKIRef;

// Returns the trust server workloop
dispatch_queue_t SecTrustServerGetWorkloop(void);

// Convert a trusted CT log array to a trusted CT log dictionary, indexed by the LogID
CF_RETURNS_RETAINED
CFDictionaryRef SecOTAPKICreateTrustedCTLogsDictionaryFromArray(CFArrayRef trustedCTLogsArray);

// Get a reference to the current OTA PKI asset data
// Caller is responsible for releasing the returned SecOTAPKIRef
CF_EXPORT CF_RETURNS_RETAINED
SecOTAPKIRef SecOTAPKICopyCurrentOTAPKIRef(void);

// Accessor to retrieve a copy of the current black listed key.
// Caller is responsible for releasing the returned CFSetRef
CF_EXPORT
CFSetRef SecOTAPKICopyRevokedListSet(SecOTAPKIRef otapkiRef);

// Accessor to retrieve a copy of the current gray listed key.
// Caller is responsible for releasing the returned CFSetRef
CF_EXPORT
CFSetRef SecOTAPKICopyDistrustedList(SecOTAPKIRef otapkiRef);

// Accessor to retrieve a copy of the current allow list dictionary.
// Caller is responsible for releasing the returned CFDictionaryRef
CF_EXPORT
CFDictionaryRef SecOTAPKICopyAllowList(SecOTAPKIRef otapkiRef);

// Accessor to retrieve a copy of the allow list for a specific authority key ID.
// Caller is responsible for releasing the returned CFArrayRef
CF_EXPORT
CFArrayRef SecOTAPKICopyAllowListForAuthKeyID(SecOTAPKIRef otapkiRef, CFStringRef authKeyID);

// Accessor to retrieve a copy of the current trusted certificate transparency logs.
// Caller is responsible for releasing the returned CFArrayRef
CF_EXPORT
CFDictionaryRef SecOTAPKICopyTrustedCTLogs(void);

// Accessor to retrieve the path of the current pinning list.
// Caller is responsible for releasing the returned CFURLRef
CF_EXPORT
CFURLRef SecOTAPKICopyPinningList(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the dictionary of EV Policy OIDs to Anchor digest.
// Caller is responsible for releasing the returned CFDictionaryRef
CF_EXPORT
CFDictionaryRef SecOTAPKICopyEVPolicyToAnchorMapping(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the dictionary of anchor digest to file offset.
// Caller is responsible for releasing the returned CFDictionaryRef
CF_EXPORT
CFDictionaryRef SecOTAPKICopyAnchorLookupTable(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the dictionary of normalized subject hash to
// certs and constraint info.
// Caller is responsible for releasing the returned CFDictionaryRef
CF_EXPORT
CFDictionaryRef SecOTAPKICopyConstrainedAnchorLookupTable(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the data of an anchor certificate specified by
// its SHA256 hash as a hex string. This retrieves the contents of a file
// in the Anchors directory whose name is <anchorHash>.cer.
// Caller is responsible for releasing the returned CFDataRef
CF_EXPORT
CFDataRef SecOTAPKICopyConstrainedAnchorData(SecOTAPKIRef otapkiRef, CFStringRef anchorHash);

// Accessor to retrieve the pointer to the top of the anchor certs file.
// Caller should NOT free the returned pointer.  The caller should hold
// a reference to the SecOTAPKIRef object until finished with
// the returned pointer.
CF_EXPORT
const char* SecOTAPKIGetAnchorTable(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the full path to the valid update snapshot resource.
// The return value may be NULL if the resource does not exist.
// Caller should NOT free the returned pointer.  The caller should hold
// a reference to the SecOTAPKIRef object until finished with
// the returned pointer.
CF_EXPORT
const char* SecOTAPKIGetValidUpdateSnapshot(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the full path to the valid database snapshot resource.
// The return value may be NULL if the resource does not exist.
// Caller should NOT free the returned pointer.  The caller should hold
// a reference to the SecOTAPKIRef object until finished with
// the returned pointer.
CF_EXPORT
const char* SecOTAPKIGetValidDatabaseSnapshot(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the current valid snapshot version.
CF_EXPORT
CFIndex SecOTAPKIGetValidSnapshotVersion(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the current valid snapshot format.
CF_EXPORT
CFIndex SecOTAPKIGetValidSnapshotFormat(SecOTAPKIRef otapkiRef);

CF_EXPORT
CFIndex SecOTAPKIGetValidSnapshotGeneration(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the OTAPKI trust store version
// Note: Trust store is not mutable by assets
CF_EXPORT
uint64_t SecOTAPKIGetTrustStoreVersion(SecOTAPKIRef otapkiRef);

// Accessor to retrieve the OTAPKI asset version
CF_EXPORT
uint64_t SecOTAPKIGetAssetVersion(SecOTAPKIRef otapkiRef);

// Accessors to retrieve the last check in time for the OTAPKI asset
CF_EXPORT
CFDateRef SecOTAPKICopyLastAssetCheckInDate(void);

#define kSecOTAPKIAssetStalenessAtRisk (60*60*24*30) // 30 days
#define kSecOTAPKIAssetStalenessWarning (60*60*24*45) // 45 days
#define kSecOTAPKIAssetStalenessDisable (60*60*24*60) // 60 days
bool SecOTAPKIAssetStalenessLessThanSeconds(CFTimeInterval seconds);

#if __OBJC__
// SPI to return the current sampling rate for the event name
// This rate is actually n where we sample 1 out of every n
NSNumber *SecOTAPKIGetSamplingRateForEvent(NSString *eventName);
#endif // __OBJC__

CFArrayRef SecOTAPKICopyAppleCertificateAuthorities(void);

extern const CFStringRef kOTAPKIKillSwitchCT;
extern const CFStringRef kOTAPKIKillSwitchNonTLSCT;
bool SecOTAPKIKillSwitchEnabled(CFStringRef switchKey);

// SPI to return the array of currently (TLS) trusted CT logs
CF_EXPORT
CFDictionaryRef SecOTAPKICopyCurrentTrustedCTLogs(CFErrorRef* error);

// SPI to return the array of currently non-TLS trusted CT logs
CF_EXPORT
CFDictionaryRef SecOTAPKICopyNonTlsTrustedCTLogs(void);

// SPI to return dictionary of CT log matching specified key id */
CF_EXPORT
CFDictionaryRef SecOTAPKICopyCTLogForKeyID(CFDataRef keyID, CFErrorRef* error);

// SPI to return the current OTA PKI trust store (PKITrustStore) asset version as a string
CF_EXPORT
CFStringRef SecOTAPKICopyCurrentTrustStoreAssetVersion(CFErrorRef* CF_RETURNS_RETAINED error);

// SPI to return the current OTA PKI trust store (PKITrustStore) content digest as a hex string
CF_EXPORT
CFStringRef SecOTAPKICopyCurrentTrustStoreContentDigest(CFErrorRef* CF_RETURNS_RETAINED error);

// SPI to check whether the given path is on an AuthAPFS volume
CF_EXPORT
bool SecOTAPKIPathIsOnAuthAPFSVolume(CFStringRef path);

// SPI to return the OTA PKI trust store version (PKITrustStore) available at the given path
// (this may be a different path than the current SecOTAPKI instance is using)
CF_EXPORT
uint64_t SecOTAPKIGetAvailableTrustStoreVersion(CFStringRef path, CFErrorRef* CF_RETURNS_RETAINED error);

// SPI to return the built-in (system) OTA PKI trust store version
// (note: some variants may not have a built-in trust store)
CF_EXPORT
uint64_t SecOTAPKIGetSystemTrustStoreVersion(CFErrorRef* CF_RETURNS_RETAINED error);

// SPI to return the current OTA PKI trust store version
// (version maintained as part of the current SecOTAPKI instance)
CF_EXPORT
uint64_t SecOTAPKIGetCurrentTrustStoreVersion(CFErrorRef* CF_RETURNS_RETAINED error);

// SPI to return the current OTA PKI (PKITrustSupplementals) asset version
CF_EXPORT
uint64_t SecOTATrustSupplementalsGetCurrentAssetVersion(CFErrorRef* error);

// SPI to return the current OTA SecExperiment asset version
CF_EXPORT
uint64_t SecOTASecExperimentGetCurrentAssetVersion(CFErrorRef* error);

// SPI to reset the current OTA PKI asset version to the version shipped
// with the system
CF_EXPORT
uint64_t SecOTAPKIResetCurrentSupplementalsAssetVersion(CFErrorRef* CF_RETURNS_RETAINED error);

// SPI to signal trustd to get a new set of trust data
// Always returns the current asset version. Returns an error with
// a reason if the update was not successful.
CF_EXPORT
uint64_t SecOTAPKISignalNewSupplementalsAsset(CFErrorRef* CF_RETURNS_RETAINED error);

// SPI to signal trustd to get a new set of SecExperiment data
// Always returns the current asset version. Returns an error with
// a reason if the update was not successful.
CF_EXPORT
uint64_t SecOTASecExperimentGetNewAsset(CFErrorRef* error);

// SPI to copy current SecExperiment asset data
CF_EXPORT
CFDictionaryRef SecOTASecExperimentCopyAsset(CFErrorRef* error);

/* "Internal" interfaces for tests */
#if __OBJC__
BOOL UpdateOTACheckInDate(void);
void UpdateKillSwitch(NSString *key, bool value);
#endif

__END_DECLS

#endif /* _OTATRUSTUTILITIES_H_ */
