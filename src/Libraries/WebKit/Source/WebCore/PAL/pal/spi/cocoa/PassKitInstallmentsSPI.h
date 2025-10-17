/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#ifndef PAL_PASSKIT_SPI_GUARD_AGAINST_INDIRECT_INCLUSION
#error "Please #include <pal/spi/cocoa/PassKitSPI.h> instead of this file directly."
#endif

#if HAVE(PASSKIT_INSTALLMENTS)

#if USE(APPLE_INTERNAL_SDK)

#if HAVE(PASSKIT_MODULARIZATION)
#import <PassKitCore/PKPaymentRequest_Private.h>
#else
#import <PassKit/PKPaymentRequest_Private.h>
#endif

#import <PassKitCore/PKPaymentMethod_Private.h>
#import <PassKitCore/PKPaymentRequestStatus_Private.h>
#import <PassKitCore/PKPayment_Private.h>

#else // !USE(APPLE_INTERNAL_SDK)

typedef NS_ENUM(NSInteger, PKInstallmentItemType) {
    PKInstallmentItemTypeGeneric = 0,
    PKInstallmentItemTypePhone,
    PKInstallmentItemTypePad,
    PKInstallmentItemTypeWatch,
    PKInstallmentItemTypeMac,
};

typedef NS_ENUM(NSInteger, PKInstallmentRetailChannel) {
    PKInstallmentRetailChannelUnknown = 0,
    PKInstallmentRetailChannelApp,
    PKInstallmentRetailChannelWeb,
    PKInstallmentRetailChannelInStore,
};

typedef NS_ENUM(NSUInteger, PKPaymentRequestType) {
    PKPaymentRequestTypeInstallment = 5,
};

@interface PKPaymentInstallmentConfiguration : NSObject <NSSecureCoding>
@end

@interface PKPaymentInstallmentItem : NSObject <NSSecureCoding>
@end

@interface PKPayment () <NSSecureCoding>
@property (nonatomic, copy) NSString *installmentAuthorizationToken;
@end

@interface PKPaymentInstallmentConfiguration ()
@property (nonatomic, assign) PKPaymentSetupFeatureType feature;
@property (nonatomic, copy) NSData *merchandisingImageData;
@property (nonatomic, strong) NSDecimalNumber *openToBuyThresholdAmount;
@property (nonatomic, strong) NSDecimalNumber *bindingTotalAmount;
@property (nonatomic, copy) NSString *currencyCode;
@property (nonatomic, assign, getter=isInStorePurchase) BOOL inStorePurchase;
@property (nonatomic, copy) NSString *installmentMerchantIdentifier;
@property (nonatomic, copy) NSString *referrerIdentifier;
@property (nonatomic, copy) NSArray<PKPaymentInstallmentItem *> *installmentItems;
@property (nonatomic, copy) NSDictionary<NSString *, id> *applicationMetadata;
@property (nonatomic, assign) PKInstallmentRetailChannel retailChannel;
@end

@interface PKPaymentInstallmentItem ()
@property (nonatomic, assign) PKInstallmentItemType installmentItemType;
@property (nonatomic, copy) NSDecimalNumber *amount;
@property (nonatomic, copy) NSString *currencyCode;
@property (nonatomic, copy) NSString *programIdentifier;
@property (nonatomic, copy) NSDecimalNumber *apr;
@property (nonatomic, copy) NSString *programTerms;
@end

@interface PKPaymentMethod () <NSSecureCoding>
@property (nonatomic, copy) NSString *bindToken;
@end

@interface PKPaymentRequest ()
@property (nonatomic, assign) PKPaymentRequestAPIType APIType;
@property (nonatomic, strong) PKPaymentInstallmentConfiguration *installmentConfiguration;
@property (nonatomic, assign) PKPaymentRequestType requestType;
@end

@interface PKPaymentRequestPaymentMethodUpdate ()
@property (nonatomic, copy) NSString *installmentGroupIdentifier;
@end

// FIXME: The SPIs above can be declared by WebKit without causing redeclaration errors on Catalina
// internal SDKs because we can avoid including the SPIs' private headers from PassKit, but we can't
// avoid importing PKPaymentSetupFeature.h due to how many other private headers include it. To avoid
// redeclaration errors while continuing to support all Catalina SDKs, declare -supportedOptions
// only when building against an internal SDK without PKPaymentInstallmentConfiguration.h (so that we
// can implement a -respondsToSelector: check). The __has_include portion of this check can be
// removed once the minimum supported Catalina internal SDK is known to contain this private header.
#if !__has_include(<PassKitCore/PKPaymentInstallmentConfiguration.h>)

typedef NS_OPTIONS(NSInteger, PKPaymentSetupFeatureSupportedOptions) {
    PKPaymentSetupFeatureSupportedOptionsInstallments = 1 << 0,
};

@interface PKPaymentSetupFeature ()
@property (nonatomic, assign, readonly) PKPaymentSetupFeatureSupportedOptions supportedOptions;
@end

#endif // !__has_include(<PassKitCore/PKPaymentInstallmentConfiguration.h>)

#endif // !USE(APPLE_INTERNAL_SDK)

#endif // HAVE(PASSKIT_INSTALLMENTS)
