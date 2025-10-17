/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#include "WebVTTTokenizer.h"

#if ENABLE(VIDEO)

#include "CSSTokenizerInputStream.h"
#include "HTMLEntityParser.h"
#include "MarkupTokenizerInlines.h"
#include <wtf/text/StringBuilder.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

#define WEBVTT_ADVANCE_TO(stateName)                        \
    do {                                                    \
        ASSERT(!m_input.isEmpty());                         \
        m_preprocessor.advance(m_input);                    \
        character = m_preprocessor.nextInputCharacter();    \
        goto stateName;                                     \
    } while (false)
#define WEBVTT_SWITCH_TO(stateName)                         \
    do { \
        ASSERT(!m_input.isEmpty()); \
        m_preprocessor.peek(m_input); \
        character = m_preprocessor.nextInputCharacter(); \
        goto stateName; \
    } while (false)

static void addNewClass(StringBuilder& classes, const StringBuilder& newClass)
{
    if (!classes.isEmpty())
        classes.append(' ');
    classes.append(newClass);
}

inline bool emitToken(WebVTTToken& resultToken, const WebVTTToken& token)
{
    resultToken = token;
    return true;
}

inline bool advanceAndEmitToken(SegmentedString& source, WebVTTToken& resultToken, const WebVTTToken& token)
{
    source.advance();
    return emitToken(resultToken, token);
}

WebVTTTokenizer::WebVTTTokenizer(const String& input)
    : m_input(input)
    , m_preprocessor(*this)
{
    // Append an EOF marker and close the input "stream".
    ASSERT(!m_input.isClosed());
    m_input.append(span(kEndOfFileMarker));
    m_input.close();
}

static void ProcessEntity(SegmentedString& source, StringBuilder& result, UChar additionalAllowedCharacter = 0)
{
    auto decoded = consumeHTMLEntity(source, additionalAllowedCharacter);
    if (decoded.failed() || decoded.notEnoughCharacters())
        result.append('&');
    else {
        for (auto character : decoded.span())
            result.append(character);
    }
}

bool WebVTTTokenizer::nextToken(WebVTTToken& token)
{
    if (m_input.isEmpty() || !m_preprocessor.peek(m_input))
        return false;

    UChar character = m_preprocessor.nextInputCharacter();
    if (character == kEndOfFileMarker) {
        m_preprocessor.advance(m_input);
        return false;
    }

    StringBuilder buffer;
    StringBuilder result;
    StringBuilder classes;

// 4.8.10.13.4 WebVTT cue text tokenizer
DataState:
    if (character == '&') {
        WEBVTT_ADVANCE_TO(HTMLCharacterReferenceInDataState);
    } else if (character == '<') {
        if (result.isEmpty())
            WEBVTT_ADVANCE_TO(TagState);
        else {
            // We don't want to advance input or perform a state transition - just return a (new) token.
            // (On the next call to nextToken we will see '<' again, but take the other branch in this if instead.)
            return emitToken(token, WebVTTToken::StringToken(result.toString()));
        }
    } else if (character == kEndOfFileMarker)
        return advanceAndEmitToken(m_input, token, WebVTTToken::StringToken(result.toString()));
    else {
        result.append(character);
        WEBVTT_ADVANCE_TO(DataState);
    }

TagState:
    if (isTokenizerWhitespace(character)) {
        ASSERT(result.isEmpty());
        WEBVTT_ADVANCE_TO(StartTagAnnotationState);
    } else if (character == '.') {
        ASSERT(result.isEmpty());
        WEBVTT_ADVANCE_TO(StartTagClassState);
    } else if (character == '/') {
        WEBVTT_ADVANCE_TO(EndTagState);
    } else if (isASCIIDigit(character)) {
        result.append(character);
        WEBVTT_ADVANCE_TO(TimestampTagState);
    } else if (character == '>' || character == kEndOfFileMarker) {
        ASSERT(result.isEmpty());
        return advanceAndEmitToken(m_input, token, WebVTTToken::StartTag(result.toString()));
    } else {
        result.append(character);
        WEBVTT_ADVANCE_TO(StartTagState);
    }

StartTagState:
    if (isTokenizerWhitespace(character))
        WEBVTT_ADVANCE_TO(StartTagAnnotationState);
    else if (character == '.')
        WEBVTT_ADVANCE_TO(StartTagClassState);
    else if (character == '>' || character == kEndOfFileMarker)
        return advanceAndEmitToken(m_input, token, WebVTTToken::StartTag(result.toString()));
    else {
        result.append(character);
        WEBVTT_ADVANCE_TO(StartTagState);
    }

StartTagClassState:
    if (isTokenizerWhitespace(character)) {
        addNewClass(classes, buffer);
        buffer.clear();
        WEBVTT_ADVANCE_TO(StartTagAnnotationState);
    } else if (character == '.') {
        addNewClass(classes, buffer);
        buffer.clear();
        WEBVTT_ADVANCE_TO(StartTagClassState);
    } else if (character == '>' || character == kEndOfFileMarker) {
        addNewClass(classes, buffer);
        buffer.clear();
        return advanceAndEmitToken(m_input, token, WebVTTToken::StartTag(result.toString(), classes.toAtomString()));
    } else {
        buffer.append(character);
        WEBVTT_ADVANCE_TO(StartTagClassState);
    }

StartTagAnnotationState:
    if (character == '&')
        WEBVTT_ADVANCE_TO(HTMLCharacterReferenceInAnnotationState);
    else if (character == '>' || character == kEndOfFileMarker)
        return advanceAndEmitToken(m_input, token, WebVTTToken::StartTag(result.toString(), classes.toAtomString(), buffer.toAtomString()));
    buffer.append(character);
    WEBVTT_ADVANCE_TO(StartTagAnnotationState);

EndTagState:
    if (character == '>' || character == kEndOfFileMarker)
        return advanceAndEmitToken(m_input, token, WebVTTToken::EndTag(result.toString()));
    result.append(character);
    WEBVTT_ADVANCE_TO(EndTagState);

TimestampTagState:
    if (character == '>' || character == kEndOfFileMarker)
        return advanceAndEmitToken(m_input, token, WebVTTToken::TimestampTag(result.toString()));
    result.append(character);
    WEBVTT_ADVANCE_TO(TimestampTagState);

HTMLCharacterReferenceInDataState:
    ProcessEntity(m_input, result);
    WEBVTT_SWITCH_TO(DataState);

HTMLCharacterReferenceInAnnotationState:
    ProcessEntity(m_input, result, '>');
    WEBVTT_SWITCH_TO(StartTagAnnotationState);
}

}

#endif
