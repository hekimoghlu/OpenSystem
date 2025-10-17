/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#include "SpeechRecognitionError.h"
#include "SpeechRecognitionResultData.h"
#include <variant>
#include <wtf/ArgumentCoder.h>

namespace WebCore {

enum class SpeechRecognitionUpdateType : uint8_t {
    Start,
    AudioStart,
    SoundStart,
    SpeechStart,
    SpeechEnd,
    SoundEnd,
    AudioEnd,
    Result,
    NoMatch,
    Error,
    End
};

String convertEnumerationToString(SpeechRecognitionUpdateType);

class SpeechRecognitionUpdate {
public:
    WEBCORE_EXPORT static SpeechRecognitionUpdate create(SpeechRecognitionConnectionClientIdentifier, SpeechRecognitionUpdateType);
    WEBCORE_EXPORT static SpeechRecognitionUpdate createError(SpeechRecognitionConnectionClientIdentifier, const SpeechRecognitionError&);
    WEBCORE_EXPORT static SpeechRecognitionUpdate createResult(SpeechRecognitionConnectionClientIdentifier, const Vector<SpeechRecognitionResultData>&);

    SpeechRecognitionConnectionClientIdentifier clientIdentifier() const { return m_clientIdentifier; }
    SpeechRecognitionUpdateType type() const { return m_type; }
    WEBCORE_EXPORT SpeechRecognitionError error() const;
    WEBCORE_EXPORT Vector<SpeechRecognitionResultData> result() const;

private:
    friend struct IPC::ArgumentCoder<SpeechRecognitionUpdate, void>;
    using Content = std::variant<std::monostate, SpeechRecognitionError, Vector<SpeechRecognitionResultData>>;
    WEBCORE_EXPORT SpeechRecognitionUpdate(SpeechRecognitionConnectionClientIdentifier, SpeechRecognitionUpdateType, Content);

    SpeechRecognitionConnectionClientIdentifier m_clientIdentifier;
    SpeechRecognitionUpdateType m_type;
    Content m_content;
};


} // namespace WebCore

namespace WTF {

template<typename> struct LogArgument;

template<> struct LogArgument<WebCore::SpeechRecognitionUpdateType> {
    static String toString(const WebCore::SpeechRecognitionUpdateType type)
    {
        return convertEnumerationToString(type);
    }
};

} // namespace WTF
