/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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

#if PLATFORM(COCOA)

#include "ArgumentCodersCocoa.h"
#include <CoreGraphics/CGGeometry.h>
#include <wtf/RetainPtr.h>

OBJC_CLASS NSValue;

namespace WebKit {

class CoreIPCNSCFObject;

class CoreIPCNSValue {
public:
    CoreIPCNSValue(NSValue *);
    CoreIPCNSValue(const RetainPtr<NSValue>& value)
        : CoreIPCNSValue(value.get()) { }
    CoreIPCNSValue(CoreIPCNSValue&&);
    ~CoreIPCNSValue();

    RetainPtr<id> toID() const;

    static bool shouldWrapValue(NSValue *);

    using Value = std::variant<NSRange, CGRect, UniqueRef<CoreIPCNSCFObject>>;

private:
    friend struct IPC::ArgumentCoder<CoreIPCNSValue, void>;

    static Value valueFromNSValue(NSValue *);

    CoreIPCNSValue(Value&&);

    Value m_value;
};

} // namespace WebKit

#endif // PLATFORM(COCOA)
