/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#import "CoreIPCNSCFObject.h"

#if PLATFORM(COCOA)

#import "ArgumentCodersCocoa.h"
#import "CoreIPCTypes.h"
#import "GeneratedWebKitSecureCoding.h"
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebKit {

static ObjectValue valueFromID(id object)
{
    if (!object)
        return nullptr;

    switch (IPC::typeFromObject(object)) {
    case IPC::NSType::Array:
        return CoreIPCArray((NSArray *)object);
    case IPC::NSType::Color:
        return CoreIPCColor((WebCore::CocoaColor *)object);
#if USE(PASSKIT)
    case IPC::NSType::PKPaymentMethod:
        return CoreIPCPKPaymentMethod((PKPaymentMethod *)object);
    case IPC::NSType::PKPaymentMerchantSession:
        return CoreIPCPKPaymentMerchantSession((PKPaymentMerchantSession *)object);
    case IPC::NSType::PKPaymentSetupFeature:
        return CoreIPCPKPaymentSetupFeature((PKPaymentSetupFeature *)object);
    case IPC::NSType::PKContact:
        return CoreIPCPKContact((PKContact *)object);
    case IPC::NSType::PKSecureElementPass:
        return CoreIPCPKSecureElementPass((PKSecureElementPass *)object);
    case IPC::NSType::PKPayment:
        return CoreIPCPKPayment((PKPayment *)object);
    case IPC::NSType::PKPaymentToken:
        return CoreIPCPKPaymentToken((PKPaymentToken *)object);
    case IPC::NSType::PKShippingMethod:
        return CoreIPCPKShippingMethod((PKShippingMethod *)object);
    case IPC::NSType::PKDateComponentsRange:
        return CoreIPCPKDateComponentsRange((PKDateComponentsRange *)object);
    case IPC::NSType::CNContact:
        return CoreIPCCNContact((CNContact *)object);
    case IPC::NSType::CNPhoneNumber:
        return CoreIPCCNPhoneNumber((CNPhoneNumber *)object);
    case IPC::NSType::CNPostalAddress:
        return CoreIPCCNPostalAddress((CNPostalAddress *)object);
#endif
#if ENABLE(DATA_DETECTION) && HAVE(WK_SECURE_CODING_DATA_DETECTORS)
    case IPC::NSType::DDScannerResult:
        return CoreIPCDDScannerResult((DDScannerResult *)object);
#if PLATFORM(MAC)
    case IPC::NSType::WKDDActionContext:
        return CoreIPCDDSecureActionContext((WKDDActionContext *)object);
#endif
#endif
    case IPC::NSType::NSDateComponents:
        return CoreIPCDateComponents((NSDateComponents *)object);
    case IPC::NSType::Data:
        return CoreIPCData((NSData *)object);
    case IPC::NSType::Date:
        return CoreIPCDate(bridge_cast((NSDate *)object));
    case IPC::NSType::Dictionary:
        return CoreIPCDictionary((NSDictionary *)object);
    case IPC::NSType::Error:
        return CoreIPCError((NSError *)object);
    case IPC::NSType::Locale:
        return CoreIPCLocale((NSLocale *)object);
    case IPC::NSType::Font:
        return CoreIPCFont((WebCore::CocoaFont *)object);
    case IPC::NSType::NSValue:
        return CoreIPCNSValue((NSValue *)object);
    case IPC::NSType::Number:
        return CoreIPCNumber(bridge_cast((NSNumber *)object));
    case IPC::NSType::Null:
        return CoreIPCNull((NSNull *)object);
#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
    case IPC::NSType::SecureCoding:
        return CoreIPCSecureCoding((NSObject<NSSecureCoding> *)object);
#endif
    case IPC::NSType::String:
        return CoreIPCString((NSString *)object);
    case IPC::NSType::URL:
        return CoreIPCURL((NSURL *)object);
    case IPC::NSType::CF:
        return CoreIPCCFType((CFTypeRef)object);
    case IPC::NSType::Unknown:
        break;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

CoreIPCNSCFObject::CoreIPCNSCFObject(id object)
    : m_value(makeUniqueRefWithoutFastMallocCheck<ObjectValue>(valueFromID(object)))
{
}

CoreIPCNSCFObject::CoreIPCNSCFObject(UniqueRef<ObjectValue>&& value)
    : m_value(WTFMove(value))
{
}

RetainPtr<id> CoreIPCNSCFObject::toID() const
{
    RetainPtr<id> result;

    WTF::switchOn(*m_value, [&](auto& object) {
        result = object.toID();
    }, [](std::nullptr_t) {
        // result should be nil, which is the default value initialized above.
    });

    return result;
}

bool CoreIPCNSCFObject::valueIsAllowed(IPC::Decoder& decoder, ObjectValue& value)
{
    // The Decoder always has a set of allowedClasses,
    // but we only check that set when considering SecureCoding classes
    Class objectClass;
    WTF::switchOn(value,
#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
        [&](CoreIPCSecureCoding& object) {
            objectClass = object.objectClass();
        },
#endif
        [&](auto& object) {
            objectClass = nullptr;
        }
    );

    return !objectClass || decoder.allowedClasses().contains(objectClass);
}

} // namespace WebKit

namespace IPC {

void ArgumentCoder<UniqueRef<WebKit::ObjectValue>>::encode(Encoder& encoder, const UniqueRef<WebKit::ObjectValue>& object)
{
    encoder << *object;
}

std::optional<UniqueRef<WebKit::ObjectValue>> ArgumentCoder<UniqueRef<WebKit::ObjectValue>>::decode(Decoder& decoder)
{
    auto object = decoder.decode<WebKit::ObjectValue>();
    if (!object)
        return std::nullopt;
    return makeUniqueRefWithoutFastMallocCheck<WebKit::ObjectValue>(WTFMove(*object));
}

} // namespace IPC

#endif // PLATFORM(COCOA)
