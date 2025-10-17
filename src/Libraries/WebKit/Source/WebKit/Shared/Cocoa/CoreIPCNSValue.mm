/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#import "CoreIPCNSValue.h"

#import "CoreIPCNSCFObject.h"
#import "CoreIPCTypes.h"

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WAKAppKitStubs.h>
#endif

#if PLATFORM(COCOA)

namespace WebKit {

CoreIPCNSValue::CoreIPCNSValue(CoreIPCNSValue&&) = default;

CoreIPCNSValue::~CoreIPCNSValue() = default;

CoreIPCNSValue::CoreIPCNSValue(Value&& value)
    : m_value(WTFMove(value)) { }

auto CoreIPCNSValue::valueFromNSValue(NSValue *nsValue) -> Value
{
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    if (!strcmp(nsValue.objCType, @encode(NSRange)))
        return nsValue.rangeValue;

    if (!strcmp(nsValue.objCType, @encode(CGRect)))
        return nsValue.rectValue;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    return makeUniqueRef<CoreIPCNSCFObject>(nsValue);
}

CoreIPCNSValue::CoreIPCNSValue(NSValue *value)
    : m_value(valueFromNSValue(value))
{
}

RetainPtr<id> CoreIPCNSValue::toID() const
{
    RetainPtr<id> result;

    auto nsValueFromWrapped = [](const Value& wrappedValue) {
        RetainPtr<id> result;

        WTF::switchOn(wrappedValue, [&](const NSRange& range) {
            result = [NSValue valueWithRange:range];
        }, [&](const CGRect& rect) {
            result = [NSValue valueWithRect:rect];
        }, [&](const UniqueRef<CoreIPCNSCFObject>& object) {
            result = object->toID();
        });

        return result;
    };

    return nsValueFromWrapped(m_value);
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
