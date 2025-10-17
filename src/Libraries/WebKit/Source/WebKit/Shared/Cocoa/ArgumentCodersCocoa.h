/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#pragma once

#import "ArgumentCoders.h"
#import "CoreIPCRetainPtr.h"

#if PLATFORM(COCOA)

#import "WKKeyedCoder.h"
#import <WebCore/AttributedString.h>
#import <wtf/RetainPtr.h>

#if ENABLE(DATA_DETECTION)
OBJC_CLASS DDScannerResult;
#if PLATFORM(MAC)
#if HAVE(SECURE_ACTION_CONTEXT)
OBJC_CLASS DDSecureActionContext;
using WKDDActionContext = DDSecureActionContext;
#else
OBJC_CLASS DDActionContext;
using WKDDActionContext = DDActionContext;
#endif // #if HAVE(SECURE_ACTION_CONTEXT)
#endif // #if PLATFORM(MAC)
#endif // #if ENABLE(DATA_DETECTION)

#if USE(AVFOUNDATION)
OBJC_CLASS AVOutputContext;
#endif

#if USE(PASSKIT)
OBJC_CLASS CNContact;
OBJC_CLASS CNPhoneNumber;
OBJC_CLASS CNPostalAddress;
OBJC_CLASS PKContact;
OBJC_CLASS PKDateComponentsRange;
OBJC_CLASS PKPayment;
OBJC_CLASS PKPaymentMerchantSession;
OBJC_CLASS PKPaymentSetupFeature;
OBJC_CLASS PKPaymentMethod;
OBJC_CLASS PKPaymentToken;
OBJC_CLASS PKShippingMethod;
OBJC_CLASS PKSecureElementPass;
#endif

OBJC_CLASS PlatformColor;
OBJC_CLASS NSShadow;

namespace IPC {

class StreamConnectionEncoder;

#ifdef __OBJC__

enum class NSType : uint8_t {
    Array,
    Color,
#if USE(PASSKIT)
    PKPaymentMethod,
    PKPaymentMerchantSession,
    PKPaymentSetupFeature,
    PKContact,
    PKSecureElementPass,
    PKPayment,
    PKPaymentToken,
    PKShippingMethod,
    PKDateComponentsRange,
    CNContact,
    CNPhoneNumber,
    CNPostalAddress,
#endif
#if ENABLE(DATA_DETECTION) && HAVE(WK_SECURE_CODING_DATA_DETECTORS)
    DDScannerResult,
#if PLATFORM(MAC)
    WKDDActionContext,
#endif
#endif
    NSDateComponents,
    Data,
    Date,
    Error,
    Dictionary,
    Font,
    Locale,
    Number,
    Null,
#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
    SecureCoding,
#endif
    String,
    URL,
    NSValue,
    CF,
    Unknown,
};
NSType typeFromObject(id);
bool isSerializableValue(id);

enum class CFType : uint8_t {
    CFArray,
    CFBoolean,
    CFCharacterSet,
    CFData,
    CFDate,
    CFDictionary,
    CFNull,
    CFNumber,
    CFString,
    CFURL,
    SecCertificate,
#if HAVE(SEC_ACCESS_CONTROL)
    SecAccessControl,
#endif
    SecTrust,
    CGColorSpace,
    CGColor,
    Nullptr,
    Unknown,
};
CFType typeFromCFTypeRef(CFTypeRef);

#if ENABLE(DATA_DETECTION)
template<> Class getClass<DDScannerResult>();
#if PLATFORM(MAC)
template<> Class getClass<WKDDActionContext>();
#endif
#endif
#if USE(AVFOUNDATION)
template<> Class getClass<AVOutputContext>();
#endif
#if USE(PASSKIT)
template<> Class getClass<CNContact>();
template<> Class getClass<CNPhoneNumber>();
template<> Class getClass<CNPostalAddress>();
template<> Class getClass<PKContact>();
template<> Class getClass<PKPaymentMerchantSession>();
template<> Class getClass<PKPaymentSetupFeature>();
template<> Class getClass<PKPayment>();
template<> Class getClass<PKPaymentToken>();
template<> Class getClass<PKShippingMethod>();
template<> Class getClass<PKDateComponentsRange>();
template<> Class getClass<PKPaymentMethod>();
template<> Class getClass<PKSecureElementPass>();
#endif

template<> Class getClass<PlatformColor>();
template<> Class getClass<NSShadow>();

template<typename T> void encodeObjectDirectly(Encoder&, T *);
template<typename T> void encodeObjectDirectly(Encoder&, T);
template<typename T> void encodeObjectDirectly(StreamConnectionEncoder&, T *);
template<typename T> void encodeObjectDirectly(StreamConnectionEncoder&, T);
template<typename T> std::optional<RetainPtr<id>> decodeObjectDirectlyRequiringAllowedClasses(Decoder&);

template<typename T, typename = IsObjCObject<T>> void encode(Encoder&, T *);

#if ASSERT_ENABLED

static inline bool isObjectClassAllowed(id object, const AllowedClassHashSet& allowedClasses)
{
    for (auto& allowedClass : allowedClasses) {
        if ([object isKindOfClass:allowedClass.get()])
            return true;
    }
    return false;
}

#endif // ASSERT_ENABLED

template<typename T, typename>
std::optional<RetainPtr<T>> decodeRequiringAllowedClasses(Decoder& decoder)
{
#if ASSERT_ENABLED
    auto allowedClasses = decoder.allowedClasses();
#endif
    auto result = decodeObjectDirectlyRequiringAllowedClasses<T>(decoder);
    if (!result)
        return std::nullopt;
    ASSERT(!*result || isObjectClassAllowed((*result).get(), allowedClasses));
    return { *result };
}

template<typename T, typename>
std::optional<T> decodeRequiringAllowedClasses(Decoder& decoder)
{
    auto result = decodeObjectDirectlyRequiringAllowedClasses<T>(decoder);
    if (!result)
        return std::nullopt;
    ASSERT(!*result || isObjectClassAllowed((*result).get(), decoder.allowedClasses()));
    return { *result };
}

template<typename T> struct ArgumentCoder<T *> {
    template<typename U = T, typename = IsObjCObject<U>>
    static void encode(Encoder& encoder, U *object)
    {
        encodeObjectDirectly<U>(encoder, object);
    }
};

#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
template<typename T> struct ArgumentCoder<CoreIPCRetainPtr<T>> {
    template<typename U = T>
    static void encode(Encoder& encoder, const CoreIPCRetainPtr<U>& object)
    {
        encodeObjectDirectly<U>(encoder, object.get());
    }

    template<typename U = T>
    static void encode(StreamConnectionEncoder& encoder, const CoreIPCRetainPtr<U>& object)
    {
        encodeObjectDirectly<U>(encoder, object.get());
    }

    template<typename U = T>
    static std::optional<RetainPtr<U>> decode(Decoder& decoder)
    {
        return decodeObjectDirectlyRequiringAllowedClasses<U>(decoder);
    }
};
#endif // !HAVE(WK_SECURE_CODING_NSURLREQUEST)

template<typename T> struct ArgumentCoder<RetainPtr<T>> {
    template<typename U = T, typename = IsObjCObject<U>>
    static void encode(Encoder& encoder, const RetainPtr<U>& object)
    {
        ArgumentCoder<U *>::encode(encoder, object.get());
    }

    template<typename U = T, typename = IsObjCObject<U>>
    static std::optional<RetainPtr<U>> decode(Decoder& decoder)
    {
        return decoder.decodeWithAllowedClasses<U>();
    }
};

template<typename T, typename>
void encode(Encoder& encoder, T *object)
{
    ArgumentCoder<T *>::encode(encoder, object);
}

#endif // __OBJC__

} // namespace IPC

#endif // PLATFORM(COCOA)
