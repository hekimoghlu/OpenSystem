/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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

#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GeolocationError : public RefCounted<GeolocationError> {
public:
    enum ErrorCode {
        PermissionDenied,
        PositionUnavailable
    };

    static Ref<GeolocationError> create(ErrorCode code, const String& message) { return adoptRef(*new GeolocationError(code, message)); }

    ErrorCode code() const { return m_code; }
    const String& message() const { return m_message; }

private:
    GeolocationError(ErrorCode code, const String& message)
        : m_code(code)
        , m_message(message)
    {
    }

    ErrorCode m_code;
    String m_message; 
};

} // namespace WebCore
