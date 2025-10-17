/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#if HAVE(CONTACTS)

#import <Contacts/Contacts.h>

#if USE(APPLE_INTERNAL_SDK)

#import <Contacts/CNContact_ReallyPrivate.h>
#import <Contacts/CNLabeledValue_Private.h>
#import <Contacts/CNMutablePostalAddress_Private.h>
#import <Contacts/CNPhoneNumber_Private.h>
#import <Contacts/CNPostalAddress_Private.h>

#else // USE(APPLE_INTERNAL_SDK)

@interface CNPhoneNumber ()
+ (nonnull instancetype)phoneNumberWithDigits:(nonnull NSString *)digits countryCode:(nullable NSString *)countryCode;

@property (readonly, copy, nullable) NSString *countryCode;
@property (readonly, copy, nonnull) NSString *digits;
@end

@interface CNPostalAddress ()
@property (copy, nullable) NSString *formattedAddress;
@end

NS_ASSUME_NONNULL_BEGIN
@interface CNContact ()
- (instancetype)initWithIdentifier:(NSString *)identifier;
@end

@interface CNLabeledValue ()
- (id)initWithIdentifier:(NSString *)identifier label:(NSString *)label value:(id<NSCopying>)value;
@end
NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)
#endif // HAVE(CONTACTS)

