/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
#ifndef OTDefines_h
#define OTDefines_h
#include <Foundation/Foundation.h>
#include "utilities/debugging.h"


#if OCTAGON
#import "keychain/ot/OTConstants.h"
#endif // OCTAGON

NS_ASSUME_NONNULL_BEGIN

extern NSString* const OTCKContainerName;

#if OCTAGON

extern NSString* const OctagonEventAttributeZoneName;
extern NSString* const OctagonEventAttributeFailureReason;
extern NSString* const OctagonEventAttributeTimeSinceLastPostedFollowUp;

extern NSString* const CuttlefishTrustZone;

/* Octagon Errors, preserved here to include all no-longer-present error code */
#if 0
typedef NS_ERROR_ENUM(OctagonErrorDomain, OctagonError) {
    //OTErrorNoColumn                         = 1,
    //OTErrorKeyGeneration                    = 2,
    //OTErrorEmptySecret                      = 3,
    //OTErrorEmptyDSID                        = 4,
    OTErrorNoIdentity                       = 5,
    //OTErrorRestoreFailed                    = 6,
    //OTErrorRestoredPeerEncryptionKeyFailure = 7,
    //OTErrorRestoredPeerSigningKeyFailure    = 8,
    //OTErrorEntropyCreationFailure           = 9,
    OTErrorDeserializationFailure           = 10,
    //OTErrorDecryptFailure                   = 11,
    //OTErrorPrivateKeyFailure                = 12,
    //OTErrorEscrowSigningSPKI                = 13,
    //OTErrorBottleID                         = 14,
    //OTErrorOTLocalStore                     = 15,
    //OTErrorOTCloudStore                     = 16,
    //OTErrorEmptyEscrowRecordID              = 17,
    //OTErrorNoBottlePeerRecords              = 18,
    //OTErrorCoreFollowUp                     = 19,
    OTErrorFeatureNotEnabled                = 20,
    OTErrorCKCallback                       = 21,
    //OTErrorRampInit                         = 22,
    OTErrorCKTimeOut                        = 23,
    OTErrorNoNetwork                        = 24,
    OTErrorNotSignedIn                      = 25,
    OTErrorRecordNotFound                   = 26,
    //OTErrorNoEscrowKeys                     = 27,
    //OTErrorBottleUpdate                     = 28,
    OTErrorNotSupported                     = 29,
    OTErrorUnexpectedStateTransition        = 30,
    OTErrorNoSuchContext                    = 31,
    //OTErrorTimeout                          = 32,
    //OTErrorMasterKey                        = 33,
    //OTErrorNotTrusted                       = 34,
    OTErrorLimitedPeer                      = 35,
    //OTErrorNoOctagonKeysInSOS               = 36,
    //OTErrorNeedsAuthentication              = 37,
    OTErrorOctagonAdapter                   = 38,
    OTErrorSOSAdapter                       = 39,
    //OctagonErrorNoAccount                   = 40,
    OTErrorRecoveryKeyMalformed             = 41,
    //OTAuthKitNoAltDSID                      = 42,
    OTErrorAuthKitAKDeviceListRequestContextClass = 43,
    OTErrorAuthKitNoPrimaryAccount          = 44,
    OTErrorAuthKitNoAuthenticationController = 45,
    OTErrorAuthKitMachineIDMissing          = 46,
    OTErrorAuthKitPrimaryAccountHaveNoDSID  = 47,
    OTErrorFailedToLeaveClique              = 48,
    OTErrorSyncPolicyMissing                = 49,
    OTErrorRequiredLibrariesNotPresent      = 50,
    OTErrorFailedToSetWalrus                = 51,
    OTErrorFailedToSetWebAccess             = 52,
    OTErrorNoAccountSettingsSet             = 53,
    OTErrorBadUUID                          = 54,
    OTErrorUserControllableViewsUnavailable = 55,
};
#endif /* 0 */

#define OTMasterSecretLength 72

typedef NS_ENUM(NSInteger, TrustedPeersHelperErrorCode) {
    TrustedPeersHelperErrorCodeNoPreparedIdentity = 1,
    TrustedPeersHelperErrorCodeUnknownPolicyVersion = 12,
    TrustedPeersHelperErrorCodeNoPeersPreapprovePreparedIdentity = 14,
    TrustedPeersHelperErrorCodeFailedToCreateRecoveryKey    = 31,
    TrustedPeersHelperErrorCodeUntrustedRecoveryKeys    = 32,
    TrustedPeersHelperErrorCodeNotEnrolled   = 34,
    TrustedPeersHelperErrorCodeUnknownCloudKitError   = 36,
    TrustedPeersHelperErrorCodeNoPeersPreapprovedBySelf = 47,
    TrustedPeersHelperErrorCodeRecoveryKeyIsNotCorrect = 52,
};

// See cuttlefish/CuttlefishService/Sources/CuttlefishService/CuttlefishError.swift
typedef NS_ENUM(NSInteger, CuttlefishErrorCode) {
    CuttlefishErrorEstablishFailed = 1001,
    CuttlefishErrorJoinFailed = 1002,
    CuttlefishErrorUpdateTrustFailed = 1004,
    CuttlefishErrorInvalidChangeToken = 1005,
    CuttlefishErrorMalformedRecord = 1006,
    CuttlefishErrorResultGraphNotFullyReachable = 1007,
    CuttlefishErrorResultGraphHasNoPotentiallyTrustedPeers = 1008,
    CuttlefishErrorResultGraphHasSplitKnowledge = 1009,
    CuttlefishErrorResultGraphHasPeerWithNoSelf = 1010,
    CuttlefishErrorInvalidEscrowProxyOperation = 1011,
    CuttlefishErrorRecordWrongType = 1012,
    CuttlefishErrorMissingMandatoryField = 1013,
    CuttlefishErrorMalformedViewKeyHierarchy = 1014,
    CuttlefishErrorUnknownView = 1015,
    CuttlefishErrorEstablishPeerFailed = 1016,
    CuttlefishErrorEstablishBottleFailed = 1017,
    CuttlefishErrorChangeTokenExpired = 1018,
    CuttlefishErrorTransactionalFailure = 1019,
    CuttlefishErrorSetRecoveryKeyFailed = 1020,
    CuttlefishErrorRetryableServerFailure = 1021,
    CuttlefishErrorPreflightGraphValidationError = 1022,
    CuttlefishErrorKeyHierarchyAlreadyExists = 1033,
    CuttlefishErrorDuplicatePeerIdUnderConsideration = 1034,
    CuttlefishErrorIneligibleExclusionDenied = 1035,
    CuttlefishErrorMultiplePreapprovedJoinDenied = 1036,
    CuttlefishErrorUpdateTrustPeerNotFound = 1037,
    CuttlefishErrorEscrowProxyFailure = 1038,
    CuttlefishErrorResetFailed = 1039,
    CuttlefishErrorViewZoneDeletionFailed = 1040,
    CuttlefishErrorAddCustodianRecoveryKeyFailed = 1041,
    CuttlefishErrorResultGraphHasNoPotentiallyTrustedPeersWithRecoveryKey = 1042,

    // For testing error handling. Never returned from actual cuttlefish.
    // Should not be retried.
    CuttlefishErrorTestGeneratedFailure = 9999,
};

#endif // OCTAGON

NS_ASSUME_NONNULL_END

#endif /* OTDefines_h */
