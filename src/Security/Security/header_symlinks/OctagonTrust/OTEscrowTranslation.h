/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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
#ifndef EscrowTranslation_h
#define EscrowTranslation_h
#if __OBJC2__

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <OctagonTrust/OTEscrowRecord.h>
#import <OctagonTrust/OTEscrowAuthenticationInformation.h>
#import <OctagonTrust/OTICDPRecordContext.h>
#import <OctagonTrust/OTCDPRecoveryInformation.h>

NS_ASSUME_NONNULL_BEGIN

@interface OTEscrowTranslation : NSObject

//dictionary to escrow auth
+ (OTEscrowAuthenticationInformation* _Nullable )dictionaryToEscrowAuthenticationInfo:(NSDictionary*)dictionary;

//escrow auth to dictionary
+ (NSDictionary* _Nullable)escrowAuthenticationInfoToDictionary:(OTEscrowAuthenticationInformation*)escrowAuthInfo;

//dictionary to escrow record
+ (OTEscrowRecord* _Nullable)dictionaryToEscrowRecord:(NSDictionary*)dictionary;

//escrow record to dictionary
+ (NSDictionary* _Nullable)escrowRecordToDictionary:(OTEscrowRecord*)escrowRecord;

//dictionary to icdp record context
+ (OTICDPRecordContext* _Nullable)dictionaryToCDPRecordContext:(NSDictionary*)dictionary;

//icdp record context to dictionary
+ (NSDictionary* _Nullable)CDPRecordContextToDictionary:(OTICDPRecordContext*)context;

+ (NSDictionary * _Nullable) metadataToDictionary:(OTEscrowRecordMetadata*)metadata;

+ (OTEscrowRecordMetadata * _Nullable) dictionaryToMetadata:(NSDictionary*)dictionary;

+ (BOOL)supportedRestorePath:(OTICDPRecordContext*)cdpContext;

@end

NS_ASSUME_NONNULL_END

#endif

#endif /* EscrowTranslation_h */
