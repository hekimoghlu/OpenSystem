/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#import "config.h"
#import "StringUtilities.h"

#import <wtf/SoftLinking.h>
#import <wtf/text/StringBuilder.h>

namespace WebKit {

#if ENABLE(TELEPHONE_NUMBER_DETECTION) && PLATFORM(MAC)

SOFT_LINK_PRIVATE_FRAMEWORK(PhoneNumbers);

typedef struct __CFPhoneNumber* CFPhoneNumberRef;

// These functions are declared with __attribute__((visibility ("default")))
// We currently don't have a way to soft link such functions, so we forward declare them again here.
extern "C" CFPhoneNumberRef CFPhoneNumberCreate(CFAllocatorRef, CFStringRef, CFStringRef);
SOFT_LINK(PhoneNumbers, CFPhoneNumberCreate, CFPhoneNumberRef, (CFAllocatorRef allocator, CFStringRef digits, CFStringRef countryCode), (allocator, digits, countryCode));

extern "C" CFStringRef CFPhoneNumberCopyFormattedRepresentation(CFPhoneNumberRef);
SOFT_LINK(PhoneNumbers, CFPhoneNumberCopyFormattedRepresentation, CFStringRef, (CFPhoneNumberRef phoneNumber), (phoneNumber));

extern "C" CFStringRef CFPhoneNumberCopyUnformattedRepresentation(CFPhoneNumberRef);
SOFT_LINK(PhoneNumbers, CFPhoneNumberCopyUnformattedRepresentation, CFStringRef, (CFPhoneNumberRef phoneNumber), (phoneNumber));


NSString *formattedPhoneNumberString(NSString *originalPhoneNumber)
{
    NSString *countryCode = [[[NSLocale currentLocale] objectForKey:NSLocaleCountryCode] lowercaseString];

    RetainPtr<CFPhoneNumberRef> phoneNumber = adoptCF(CFPhoneNumberCreate(kCFAllocatorDefault, (__bridge CFStringRef)originalPhoneNumber, (__bridge CFStringRef)countryCode));
    if (!phoneNumber)
        return originalPhoneNumber;

    auto phoneNumberString = adoptCF(CFPhoneNumberCopyFormattedRepresentation(phoneNumber.get()));
    if (!phoneNumberString)
        phoneNumberString = adoptCF(CFPhoneNumberCopyUnformattedRepresentation(phoneNumber.get()));

    return phoneNumberString.bridgingAutorelease();
}

#endif // ENABLE(TELEPHONE_NUMBER_DETECTION) && PLATFORM(MAC)

}
