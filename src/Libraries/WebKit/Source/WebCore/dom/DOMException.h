/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#include "ExceptionCode.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class Exception;

class DOMException : public RefCounted<DOMException> {
public:
    static Ref<DOMException> create(ExceptionCode, const String& message = emptyString());
    static Ref<DOMException> create(const Exception&);

    // For DOM bindings.
    static Ref<DOMException> create(const String& message, const String& name);

    virtual ~DOMException() { }

    using LegacyCode = uint8_t;
    LegacyCode legacyCode() const { return m_legacyCode; }

    String name() const { return m_name; }
    String message() const { return m_message; }

    struct Description {
        const ASCIILiteral name;
        const ASCIILiteral message;
        LegacyCode legacyCode;
    };

    WEBCORE_EXPORT static const Description& description(ExceptionCode);

    static ASCIILiteral name(ExceptionCode ec) { return description(ec).name; }
    static ASCIILiteral message(ExceptionCode ec) { return description(ec).message; }

protected:
    DOMException(LegacyCode, const String& name, const String& message);

private:
    LegacyCode m_legacyCode;
    String m_name;
    String m_message;
};

} // namespace WebCore
