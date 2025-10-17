/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include <JavaScriptCore/Forward.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class WebKitMediaKeyError : public RefCounted<WebKitMediaKeyError> {
public:
    enum {
        MEDIA_KEYERR_UNKNOWN = 1,
        MEDIA_KEYERR_CLIENT,
        MEDIA_KEYERR_SERVICE,
        MEDIA_KEYERR_OUTPUT,
        MEDIA_KEYERR_HARDWARECHANGE,
        MEDIA_KEYERR_DOMAIN
    };
    typedef unsigned short Code;

    static Ref<WebKitMediaKeyError> create(Code code, uint32_t systemCode = 0) { return adoptRef(*new WebKitMediaKeyError(code, systemCode)); }

    Code code() const { return m_code; }
    uint32_t systemCode() { return m_systemCode; }

private:
    explicit WebKitMediaKeyError(Code code, unsigned long systemCode) : m_code(code), m_systemCode(systemCode) { }

    Code m_code;
    unsigned long m_systemCode;
};

} // namespace WebCore

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
