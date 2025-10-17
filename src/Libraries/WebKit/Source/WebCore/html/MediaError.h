/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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

#if ENABLE(VIDEO)

#include <wtf/RefCounted.h>

namespace WebCore {

class MediaError : public RefCounted<MediaError> {
public:
    enum Code {
        MEDIA_ERR_ABORTED = 1,
        MEDIA_ERR_NETWORK,
        MEDIA_ERR_DECODE,
        MEDIA_ERR_SRC_NOT_SUPPORTED
#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
        , MEDIA_ERR_ENCRYPTED
#endif
    };

    static Ref<MediaError> create(Code code, String&& message)
    {
        return adoptRef(*new MediaError(code, WTFMove(message)));
    }

    Code code() const { return m_code; }
    const String& message() const { return m_message; }

private:
    MediaError(Code code, String&& message)
        : m_code(code)
        , m_message(WTFMove(message))
    { }

    Code m_code;
    String m_message;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
