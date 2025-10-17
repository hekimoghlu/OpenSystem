/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#ifndef OTConstants_h
#define OTConstants_h

#include <stdbool.h>

bool OctagonIsSOSFeatureEnabled(void);
bool OctagonPlatformSupportsSOS(void);
void OctagonSetSOSFeatureEnabled(bool value);
bool SOSCompatibilityModeEnabled(void);
void SetSOSCompatibilityMode(bool value);
void ClearSOSCompatibilityModeOverride(void);
bool IsRollOctagonIdentityEnabled(void);
void SetRollOctagonIdentityEnabled(bool value);
void ClearRollOctagonIdentityEnabledOverride(void);

#if __OBJC__

#import <Foundation/Foundation.h>
#import <AppleFeatures/AppleFeatures.h>

extern NSString* OTDefaultContext;

extern NSErrorDomain const OctagonErrorDomain;

typedef NS_ERROR_ENUM(OctagonErrorDomain, OctagonError) {
    OctagonErrorNoIdentity                                      = 5,
    OctagonErrorDeserializationFailure                          = 10,
    OctagonErrorFeatureNotEnabled                               = 20,
    OctagonErrorCKCallback                                      = 21,
    OctagonErrorCKTimeOut                                       = 23,
    OctagonErrorNoNetwork                                       = 24,
    OctagonErrorNotSignedIn                                     = 25,
    OctagonErrorRecordNotFound                                  = 26,
    OctagonErrorNotSupported                                    = 29,
    OctagonErrorUnexpectedStateTransition                       = 30,
    OctagonErrorNoSuchContext                                   = 31,
    OctagonErrorOperationUnavailableOnLimitedPeer               = 35,
    OctagonErrorOctagonAdapter                                  = 38,
    OctagonErrorSOSAdapter                                      = 39,
    OctagonErrorRecoveryKeyMalformed                            = 41,
    OctagonErrorAuthKitAKDeviceListRequestContextClass          = 43,
    OctagonErrorAuthKitNoPrimaryAccount                         = 44,
    OctagonErrorAuthKitNoAuthenticationController               = 45,
    OctagonErrorAuthKitMachineIDMissing                         = 46,
    OctagonErrorAuthKitPrimaryAccountHaveNoDSID                 = 47,
    OctagonErrorFailedToLeaveClique                             = 48,
    OctagonErrorSyncPolicyMissing                               = 49,
    OctagonErrorRequiredLibrariesNotPresent                     = 50,
    OctagonErrorFailedToSetWalrus                               = 51,
    OctagonErrorFailedToSetWebAccess                            = 52,
    OctagonErrorNoAccountSettingsSet                            = 53,
    OctagonErrorBadUUID                                         = 54,
    OctagonErrorUserControllableViewsUnavailable                = 55,
    OctagonErrorICloudAccountStateUnknown                       = 56,
    OctagonErrorClassCLocked                                    = 57,
    OctagonErrorRecordNotViable                                 = 58,
    OctagonErrorNoAppleAccount                                  = 59,
    OctagonErrorInvalidPersona                                  = 60,
    OctagonErrorNoSuchCKKS                                      = 61,
    OctagonErrorUnsupportedInEDUMode                            = 62,
    OctagonErrorAltDSIDPersonaMismatch                          = 63,
    OctagonErrorNoRecoveryKeyRegistered                         = 64,
    OctagonErrorRecoverWithRecoveryKeyNotSupported              = 65,
    OctagonErrorSecureBackupRestoreUsingRecoveryKeyFailed       = 66,
    OctagonErrorRecoveryKeyIncorrect                            = 67,
    OctagonErrorBadAuthKitResponse                              = 68,
    OctagonErrorUnsupportedAccount                              = 69,
    OctagonErrorSOSDisabled                                     = 70,
    OctagonErrorNotInSOS                                        = 71,
    OctagonErrorInjectedError                                   = 72,
    OctagonErrorCannotSetAccountSettings                        = 73,
    OctagonErrorInvalidPeerIDforPermanentInfo                   = 74,
    OctagonErrorInvalidPeerTypeForMaxCapability                 = 75,
    OctagonErrorCKKSLackingTrust                                = 76,
    OctagonErrorAuthKitNoAccountManager                         = 77,
    OctagonErrorFailedToRecoverWithCDPContext                   = 78,
    OctagonErrorFailedToRecoverWithInfo                         = 79,
    OctagonErrorFailedToHandleRecoveryResults                   = 80,
};

/* used for defaults writes */
extern NSString* OTDefaultsDomain;

extern NSString* OTProtocolPairing;
extern NSString* OTProtocolPiggybacking;

extern const char * OTTrustStatusChangeNotification;
extern NSString* OTEscrowRecordPrefix;

// Used for testing.

bool OctagonSupportsPersonaMultiuser(void);
void OctagonSetSupportsPersonaMultiuser(bool value);
void OctagonClearSupportsPersonaMultiuserOverride(void);



typedef NS_ENUM(NSInteger, CuttlefishResetReason) {
    CuttlefishResetReasonUnknown = 0,
    CuttlefishResetReasonUserInitiatedReset = 1,
    CuttlefishResetReasonHealthCheck = 2,
    CuttlefishResetReasonNoBottleDuringEscrowRecovery = 3,
    CuttlefishResetReasonLegacyJoinCircle = 4,
    CuttlefishResetReasonRecoveryKey = 5,
    CuttlefishResetReasonTestGenerated = 6,
};

extern NSString* const CuttlefishErrorDomain;
extern NSString* const CuttlefishErrorRetryAfterKey;

typedef NS_ENUM(NSInteger, OTEscrowRecordFetchSource) {
    /// Default is equivalent to cache or cuttlefish, depending on recency of cache update.
    OTEscrowRecordFetchSourceDefault = 0,
    
    /// Forces the escrow record fetch to only use local on-disk cache, even if stale.
    OTEscrowRecordFetchSourceCache = 1,
    
    /// Forces the escrow record fetch to only use cuttlefish, even if cache is recent.
    OTEscrowRecordFetchSourceCuttlefish = 2,
};

extern NSString* const TrustedPeersHelperRecoveryKeySetErrorDomain;
extern NSString* const TrustedPeersHelperErrorDomain;

typedef NS_ERROR_ENUM(TrustedPeersHelperRecoveryKeySetErrorDomain, TrustedPeersHelperRecoveryKeySetErrorCode) {
    TrustedPeersHelperRecoveryKeySetErrorKeyGeneration = 1,
    TrustedPeersHelperRecoveryKeySetErrorItemDoesNotExist = 2,
    TrustedPeersHelperRecoveryKeySetErrorFailedToSaveToKeychain = 3,
    TrustedPeersHelperRecoveryKeySetErrorUnsupportedKeyType = 4,
    TrustedPeersHelperRecoveryKeySetErrorCoreCryptoKeyGeneration = 5,
    TrustedPeersHelperRecoveryKeySetErrorFailedToGenerateRandomKey = 6
};


#endif // __OBJC__

#endif /* OTConstants_h */
