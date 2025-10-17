/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
//  SecurityPairing.h
//  Security
//

#import <Foundation/Foundation.h>
#import <Security/SecureObjectSync/SOSTypes.h>

extern NSString *kKCPairingChannelErrorDomain;

#define KCPairingErrorNoControlChannel          1
#define KCPairingErrorTooManySteps              2
#define KCPairingErrorAccountCredentialMissing  3
#define KCPairingErrorOctagonMessageMissing     4
#define KCPairingErrorTypeConfusion             5
#define KCPairingErrorNoTrustAvailable          6
#define KCPairingErrorNoStashedCredential       7
#define KCPairingErrorNotTrustedInOctagon       8
#define KCPairingErrorWrongPacketFormat         9
#define KCPairingErrorUnexpectedKeychainType    10
#define KCPairingErrorUnableToFetchDSID         11

typedef void(^KCPairingChannelCompletion)(BOOL complete, NSData *packet, NSError *error);

typedef NSString* KCPairingIntent_Type NS_STRING_ENUM;
extern KCPairingIntent_Type const KCPairingIntent_Type_None;
extern KCPairingIntent_Type const KCPairingIntent_Type_SilentRepair;
extern KCPairingIntent_Type const KCPairingIntent_Type_UserDriven;

typedef NSString* KCPairingIntent_Capability NS_STRING_ENUM;
extern KCPairingIntent_Capability const KCPairingIntent_Capability_FullPeer;
extern KCPairingIntent_Capability const KCPairingIntent_Capability_LimitedPeer;

typedef NSString *PairingPacketKey NS_STRING_ENUM;
extern PairingPacketKey const PairingPacketKey_Device;
extern PairingPacketKey const PairingPacketKey_OctagonData;
extern PairingPacketKey const PairingPacketKey_PeerJoinBlob;
extern PairingPacketKey const PairingPacketKey_Credential;
extern PairingPacketKey const PairingPacketKey_CircleBlob;
extern PairingPacketKey const PairingPacketKey_InitialCredentialsFromAcceptor;
extern PairingPacketKey const PairingPacketKey_Version;
@interface KCPairingChannelContext : NSObject <NSSecureCoding>
@property (nonatomic, copy) NSString *model;
@property (nonatomic, copy) NSString *modelVersion;
@property (nonatomic, copy) NSString *modelClass;
@property (nonatomic, copy) NSString *osVersion;
@property (nonatomic, copy) NSString *uniqueDeviceID;
@property (nonatomic, copy) NSString *uniqueClientID;
@property (nonatomic, copy) NSString *altDSID;
@property (nonatomic, copy) NSString *flowID;
@property (nonatomic, copy) NSString *deviceSessionID;
@property (nonatomic, copy) KCPairingIntent_Type intent;
@property (nonatomic, copy) KCPairingIntent_Capability capability;
@end

/**
 * Pairing channel provides the channel used in OOBE / D2D and HomePod/AppleTV (TapToFix) setup.
 *
 * The initiator is the device that wants to get into the circle, the acceptor is the device that is already in.
 * The interface require the caller to hold a lock assertion over the whole transaction, both on the initiator and acceptor.
*/

@interface KCPairingChannel : NSObject

@property (assign,readonly) BOOL needInitialSync;

@property (nonatomic, readonly, strong) KCPairingChannelContext *peerVersionContext;

+ (instancetype)pairingChannelInitiator:(KCPairingChannelContext *)peerVersionContext;
+ (instancetype)pairingChannelAcceptor:(KCPairingChannelContext *)peerVersionContext;

- (instancetype)initAsInitiator:(bool)initiator version:(KCPairingChannelContext *)peerVersionContext;
- (void)validateStart:(void(^)(bool result, NSError *error))complete;

- (NSData *)exchangePacket:(NSData *)data complete:(bool *)complete error:(NSError **)error;
- (void)exchangePacket:(NSData *)inputCompressedData complete:(KCPairingChannelCompletion)complete; /* async version of above */

/* for tests cases only */
- (void)setXPCConnectionObject:(NSXPCConnection *)connection;
- (void)setControlObject:(id)control;

- (void)setSessionControlArguments:(id)controlArguments;
- (void)setConfiguration:(id)config;
- (void)setSOSMessageFailForTesting:(BOOL)value;
- (void)setOctagonMessageFailForTesting:(BOOL)value;
+ (bool)isSupportedPlatform;
- (void)setSessionSupportsOctagonForTesting:(bool)value;
+ (NSData *)pairingChannelCompressData:(NSData *)data;
+ (NSData *)pairingChannelDecompressData:(NSData *)data;

@end

