/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#if ENABLE(WEB_AUTHN) && HAVE(NEAR_FIELD)

#if USE(APPLE_INTERNAL_SDK)

#import <NearField/NearField.h>

#else

typedef NS_OPTIONS(NSUInteger, NFFeature) {
    NFFeatureReaderMode                 = 1 << 0,
};

typedef NS_ENUM(uint32_t, NFTagType) {
    NFTagTypeUnknown        = 0,
    NFTagTypeGeneric4A      = 3,
    NFTagTypeGeneric4B      = 6,
    NFTagTypeMiFareDESFire  = 16,
};

typedef NS_OPTIONS(uint32_t, NFTechnology) {
    NFTechnologyNone                 = 0,
};

typedef NS_ENUM(uint32_t, NFNdefAvailability) {
    NFNdefAvailabilityUnknown = 0,
};

@protocol NFTagA
@end

@protocol NFTagB
@end

@protocol NFTagF
@end

@protocol NFTag <NSObject>
@property (nonatomic, readonly) NFTagType type;
@property (nonatomic, readonly) NFTechnology technology;
@property (nonatomic, readonly, copy) NSData *tagID;
@property (nonatomic, readonly, copy) NSData *UID;
@property (nonatomic, readonly, assign) NFNdefAvailability ndefAvailability;
@property (nonatomic, readonly, assign) size_t ndefMessageSize;
@property (nonatomic, readonly, assign) size_t ndefContainerSize;
@property (nonatomic, readonly, copy) NSData *AppData NS_DEPRECATED(10_12, 10_15, 10_0, 13_0);

@property (nonatomic, readonly) id<NFTagA> tagA;
@property (nonatomic, readonly) id<NFTagB> tagB;
@property (nonatomic, readonly) id<NFTagF> tagF;

- (instancetype)initWithNFTag:(id<NFTag>)tag;
- (NSString *)description;
- (BOOL)isEqualToNFTag:(id<NFTag>)tag;
@end

@interface NFTag : NSObject <NSSecureCoding, NFTag, NFTagA, NFTagB, NFTagF>
@end

@protocol NFSession <NSObject>
- (void)endSession;
@end

@interface NFSession : NSObject <NFSession>
@end

@protocol NFReaderSessionDelegate;

typedef NS_ENUM(NSInteger, NFReaderSessionUI) {
    NFReaderSessionUINone
};

@interface NFReaderSession : NFSession
@property (assign) id<NFReaderSessionDelegate> delegate;

- (instancetype)initWithUIType:(NFReaderSessionUI)uiType;
- (BOOL)startPollingWithError:(NSError **)outError;
- (BOOL)stopPolling;
- (BOOL)connectTag:(NFTag*)tag;
- (BOOL)disconnectTag;
- (NSData*)transceive:(NSData*)capdu;
@end

@protocol NFReaderSessionDelegate <NSObject>
@optional
- (void)readerSession:(NFReaderSession*)theSession didDetectTags:(NSArray<NFTag *> *)tags;
@end

@interface NFHardwareManager : NSObject
+ (instancetype)sharedHardwareManager;
- (NSObject<NFSession> *)startReaderSession:(void(^)(NFReaderSession *session, NSError *error))theStartCallback;
- (BOOL)areFeaturesSupported:(NFFeature)featureMask outError:(NSError**)outError;
@end

#endif // USE(APPLE_INTERNAL_SDK)

#endif // ENABLE(WEB_AUTHN) && HAVE(NEAR_FIELD)
