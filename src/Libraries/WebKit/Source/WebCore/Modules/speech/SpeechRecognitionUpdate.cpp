/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#include "config.h"
#include "SpeechRecognitionUpdate.h"

#include <wtf/NeverDestroyed.h>

namespace WebCore {

String convertEnumerationToString(SpeechRecognitionUpdateType enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 11> values {
        MAKE_STATIC_STRING_IMPL("UpdateTypeStart"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeAudioStart"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeSoundStart"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeSpeechStart"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeSpeechEnd"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeSoundEnd"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeAudioEnd"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeResult"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeNoMatch"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeError"),
        MAKE_STATIC_STRING_IMPL("UpdateTypeEnd"),
    };
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::Start) == 0, "SpeechRecognitionUpdateType::Start is not 1 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::AudioStart) == 1, "SpeechRecognitionUpdateType::AudioStart is not 2 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::SoundStart) == 2, "SpeechRecognitionUpdateType::SoundStart is not 3 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::SpeechStart) == 3, "SpeechRecognitionUpdateType::SpeechStart is not 4 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::SpeechEnd) == 4, "SpeechRecognitionUpdateType::SpeechEnd is not 5 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::SoundEnd) == 5, "SpeechRecognitionUpdateType::SoundEnd is not 6 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::AudioEnd) == 6, "SpeechRecognitionUpdateType::AudioEnd is not 7 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::Result) == 7, "SpeechRecognitionUpdateType::Result is not 8 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::NoMatch) == 8, "SpeechRecognitionUpdateType::NoMatch is not 9 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::Error) == 9, "SpeechRecognitionUpdateType::Error is not 10 as expected");
    static_assert(static_cast<size_t>(SpeechRecognitionUpdateType::End) == 10, "SpeechRecognitionUpdateType::End is not 11 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

SpeechRecognitionUpdate SpeechRecognitionUpdate::create(SpeechRecognitionConnectionClientIdentifier clientIdentifier, SpeechRecognitionUpdateType type)
{
    return SpeechRecognitionUpdate { clientIdentifier, type, std::monostate() };
}

SpeechRecognitionUpdate SpeechRecognitionUpdate::createError(SpeechRecognitionConnectionClientIdentifier clientIdentifier, const SpeechRecognitionError& error)
{
    return SpeechRecognitionUpdate { clientIdentifier, SpeechRecognitionUpdateType::Error, error };
}

SpeechRecognitionUpdate SpeechRecognitionUpdate::createResult(SpeechRecognitionConnectionClientIdentifier clientIdentifier, const Vector<SpeechRecognitionResultData>& results)
{
    return SpeechRecognitionUpdate { clientIdentifier, SpeechRecognitionUpdateType::Result, results };
}

SpeechRecognitionUpdate::SpeechRecognitionUpdate(SpeechRecognitionConnectionClientIdentifier clientIdentifier, SpeechRecognitionUpdateType type, Content content)
    : m_clientIdentifier(clientIdentifier)
    , m_type(type)
    , m_content(content)
{
}

SpeechRecognitionError SpeechRecognitionUpdate::error() const
{
    return WTF::switchOn(m_content,
        [] (const SpeechRecognitionError& error) { return error; },
        [] (const auto&) { return SpeechRecognitionError(); }
    );
}

Vector<SpeechRecognitionResultData> SpeechRecognitionUpdate::result() const
{
    return WTF::switchOn(m_content,
        [] (const Vector<SpeechRecognitionResultData>& data) { return data; },
        [] (const auto&) { return Vector<SpeechRecognitionResultData> { }; }
    );
}

} // namespace WebCore
