/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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

#include "CharacterRange.h"
#include "TextCheckingRequestIdentifier.h"
#include <wtf/ObjectIdentifier.h>
#include <wtf/OptionSet.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class TextCheckingType : uint8_t {
    None                    = 0,
    Spelling                = 1 << 0,
    Grammar                 = 1 << 1,
    Link                    = 1 << 2,
    Quote                   = 1 << 3,
    Dash                    = 1 << 4,
    Replacement             = 1 << 5,
    Correction              = 1 << 6,
    ShowCorrectionPanel     = 1 << 7,
};

#if PLATFORM(MAC)
typedef uint64_t NSTextCheckingTypes;
WEBCORE_EXPORT NSTextCheckingTypes nsTextCheckingTypes(OptionSet<TextCheckingType>);
#endif

enum class TextCheckingProcessType : bool {
    TextCheckingProcessBatch,
    TextCheckingProcessIncremental
};

struct GrammarDetail {
    CharacterRange range;
    Vector<String> guesses;
    String userDescription;
};

struct TextCheckingResult {
    OptionSet<TextCheckingType> type;
    CharacterRange range;
    Vector<GrammarDetail> details;
    String replacement;
};

struct TextCheckingGuesses {
    Vector<String> guesses;
    bool misspelled { false };
    bool ungrammatical { false };
};


class TextCheckingRequestData {
    friend class SpellCheckRequest; // For access to m_identifier.
public:
    TextCheckingRequestData() = default;
    TextCheckingRequestData(std::optional<TextCheckingRequestIdentifier> identifier, const String& text, OptionSet<TextCheckingType> checkingTypes, TextCheckingProcessType processType)
        : m_text { text }
        , m_identifier { identifier }
        , m_processType { processType }
        , m_checkingTypes { checkingTypes }
    {
    }

    std::optional<TextCheckingRequestIdentifier> identifier() const { return m_identifier; }
    const String& text() const { return m_text; }
    OptionSet<TextCheckingType> checkingTypes() const { return m_checkingTypes; }
    TextCheckingProcessType processType() const { return m_processType; }

private:
    String m_text;
    std::optional<TextCheckingRequestIdentifier> m_identifier;
    TextCheckingProcessType m_processType { TextCheckingProcessType::TextCheckingProcessIncremental };
    OptionSet<TextCheckingType> m_checkingTypes;
};

class TextCheckingRequest : public RefCounted<TextCheckingRequest> {
public:
    virtual ~TextCheckingRequest() = default;

    virtual const TextCheckingRequestData& data() const = 0;
    virtual void didSucceed(const Vector<TextCheckingResult>&) = 0;
    virtual void didCancel() = 0;
};

} // namespace WebCore
