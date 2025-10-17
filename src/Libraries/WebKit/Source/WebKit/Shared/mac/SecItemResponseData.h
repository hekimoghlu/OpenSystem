/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>

namespace WebKit {

class SecItemResponseData {
public:
    using Result = std::variant<
        std::nullptr_t,
        Vector<RetainPtr<SecCertificateRef>>
#if HAVE(SEC_KEYCHAIN)
        , Vector<RetainPtr<SecKeychainItemRef>>
#endif
        , RetainPtr<CFTypeRef>
    >;
    SecItemResponseData(OSStatus code, Result&& result)
        : m_resultCode(code)
        , m_resultObject(WTFMove(result)) { }

    Result& resultObject() { return m_resultObject; }
    const Result& resultObject() const { return m_resultObject; }
    OSStatus resultCode() const { return m_resultCode; }

private:
    OSStatus m_resultCode;
    Result m_resultObject;
};
    
} // namespace WebKit
