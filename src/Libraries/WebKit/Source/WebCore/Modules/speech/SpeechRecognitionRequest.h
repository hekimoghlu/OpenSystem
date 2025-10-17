/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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

#include "SpeechRecognitionConnectionClientIdentifier.h"
#include "SpeechRecognitionRequestInfo.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class SpeechRecognitionRequest;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SpeechRecognitionRequest> : std::true_type { };
}

namespace WebCore {

class SpeechRecognitionRequest : public CanMakeWeakPtr<SpeechRecognitionRequest> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SpeechRecognitionRequest, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT explicit SpeechRecognitionRequest(SpeechRecognitionRequestInfo&&);

    SpeechRecognitionConnectionClientIdentifier clientIdentifier() const { return m_info.clientIdentifier; }
    const String& lang() const { return m_info.lang; }
    bool continuous() const { return m_info.continuous;; }
    bool interimResults() const { return m_info.interimResults; }
    uint64_t maxAlternatives() const { return m_info.maxAlternatives; }
    const ClientOrigin clientOrigin() const { return m_info.clientOrigin; }
    FrameIdentifier frameIdentifier() const { return m_info.frameIdentifier; }

private:
    SpeechRecognitionRequestInfo m_info;
};

} // namespace WebCore
