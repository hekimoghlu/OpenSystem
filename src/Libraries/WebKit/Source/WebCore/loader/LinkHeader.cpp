/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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
#include "LinkHeader.h"

#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

template<typename CharacterType> static bool isNotURLTerminatingChar(CharacterType character)
{
    return character != '>';
}

template<typename CharacterType> static bool isValidParameterNameChar(CharacterType character)
{
    // A separator, CTL or '%', '*' or '\'' means the char is not valid.
    // Definition as attr-char at https://tools.ietf.org/html/rfc5987
    // CTL and separators are defined in https://tools.ietf.org/html/rfc2616#section-2.2
    // Valid chars are alpha-numeric and any of !#$&+-.^_`|~"
    if ((character >= '^' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || (character >= '!' && character <= '$') || character == '&' || character == '+' || character == '-' || character == '.')
        return true;
    return false;
}

template<typename CharacterType> static bool isParameterValueEnd(CharacterType character)
{
    return character == ';' || character == ',';
}

template<typename CharacterType> static bool isParameterValueChar(CharacterType character)
{
    return !isTabOrSpace(character) && !isParameterValueEnd(character);
}

// Verify that the parameter is a link-extension which according to spec doesn't have to have a value.
static bool isExtensionParameter(LinkHeader::LinkParameterName name)
{
    return name >= LinkHeader::LinkParameterUnknown;
}

// Before:
//
// <cat.jpg>; rel=preload
// ^                     ^
// position              end
//
// After (if successful: otherwise the method returns false)
//
// <cat.jpg>; rel=preload
//          ^            ^
//          position     end
template<typename CharacterType> static std::optional<String> findURLBoundaries(StringParsingBuffer<CharacterType>& buffer)
{
    skipWhile<isTabOrSpace>(buffer);
    if (!skipExactly(buffer, '<'))
        return std::nullopt;
    skipWhile<isTabOrSpace>(buffer);

    auto urlStart = buffer.span();
    skipWhile<isNotURLTerminatingChar>(buffer);
    auto urlEnd = buffer.position();
    skipUntil(buffer, '>');
    if (!skipExactly(buffer, '>'))
        return std::nullopt;

    return String(urlStart.first(urlEnd - urlStart.data()));
}

template<typename CharacterType> static bool invalidParameterDelimiter(StringParsingBuffer<CharacterType>& buffer)
{
    return !skipExactly(buffer, ';') && buffer.hasCharactersRemaining() && *buffer != ',';
}

template<typename CharacterType> static bool validFieldEnd(StringParsingBuffer<CharacterType>& buffer)
{
    return buffer.atEnd() || *buffer == ',';
}

// Before:
//
// <cat.jpg>; rel=preload
//          ^            ^
//          position     end
//
// After (if successful: otherwise the method returns false, and modifies the isValid boolean accordingly)
//
// <cat.jpg>; rel=preload
//            ^          ^
//            position  end
template<typename CharacterType> static bool parseParameterDelimiter(StringParsingBuffer<CharacterType>& buffer, bool& isValid)
{
    isValid = true;
    skipWhile<isTabOrSpace>(buffer);
    if (invalidParameterDelimiter(buffer)) {
        isValid = false;
        return false;
    }
    skipWhile<isTabOrSpace>(buffer);
    if (validFieldEnd(buffer))
        return false;
    return true;
}

static LinkHeader::LinkParameterName parameterNameFromString(StringView name)
{
    if (equalLettersIgnoringASCIICase(name, "rel"_s))
        return LinkHeader::LinkParameterRel;
    if (equalLettersIgnoringASCIICase(name, "anchor"_s))
        return LinkHeader::LinkParameterAnchor;
    if (equalLettersIgnoringASCIICase(name, "crossorigin"_s))
        return LinkHeader::LinkParameterCrossOrigin;
    if (equalLettersIgnoringASCIICase(name, "title"_s))
        return LinkHeader::LinkParameterTitle;
    if (equalLettersIgnoringASCIICase(name, "media"_s))
        return LinkHeader::LinkParameterMedia;
    if (equalLettersIgnoringASCIICase(name, "type"_s))
        return LinkHeader::LinkParameterType;
    if (equalLettersIgnoringASCIICase(name, "rev"_s))
        return LinkHeader::LinkParameterRev;
    if (equalLettersIgnoringASCIICase(name, "hreflang"_s))
        return LinkHeader::LinkParameterHreflang;
    if (equalLettersIgnoringASCIICase(name, "as"_s))
        return LinkHeader::LinkParameterAs;
    if (equalLettersIgnoringASCIICase(name, "imagesrcset"_s))
        return LinkHeader::LinkParameterImageSrcSet;
    if (equalLettersIgnoringASCIICase(name, "imagesizes"_s))
        return LinkHeader::LinkParameterImageSizes;
    if (equalLettersIgnoringASCIICase(name, "nonce"_s))
        return LinkHeader::LinkParameterNonce;
    if (equalLettersIgnoringASCIICase(name, "referrerpolicy"_s))
        return LinkHeader::LinkParameterReferrerPolicy;
    if (equalLettersIgnoringASCIICase(name, "fetchpriority"_s))
        return LinkHeader::LinkParameterFetchPriority;
    return LinkHeader::LinkParameterUnknown;
}

// Before:
//
// <cat.jpg>; rel=preload
//            ^          ^
//            position   end
//
// After (if successful: otherwise the method returns false)
//
// <cat.jpg>; rel=preload
//                ^      ^
//            position  end
template<typename CharacterType> static std::optional<LinkHeader::LinkParameterName> parseParameterName(StringParsingBuffer<CharacterType>& buffer)
{
    auto nameStart = buffer.span();
    skipWhile<isValidParameterNameChar>(buffer);
    auto nameEnd = buffer.position();
    skipWhile<isTabOrSpace>(buffer);
    bool hasEqual = skipExactly(buffer, '=');
    skipWhile<isTabOrSpace>(buffer);
    auto name = parameterNameFromString(nameStart.first(static_cast<size_t>(nameEnd - nameStart.data())));
    if (hasEqual)
        return name;
    bool validParameterValueEnd = buffer.atEnd() || isParameterValueEnd(*buffer);
    if (validParameterValueEnd && isExtensionParameter(name))
        return name;
    return std::nullopt;
}

// Before:
//
// <cat.jpg>; rel="preload"; type="image/jpeg";
//                ^                            ^
//            position                        end
//
// After (if the parameter starts with a quote, otherwise the method returns false)
//
// <cat.jpg>; rel="preload"; type="image/jpeg";
//                         ^                   ^
//                     position               end
template<typename CharacterType> static bool skipQuotesIfNeeded(StringParsingBuffer<CharacterType>& buffer, bool& completeQuotes)
{
    auto startSpan = buffer.span();
    unsigned char quote;
    if (skipExactly(buffer, '\''))
        quote = '\'';
    else if (skipExactly(buffer, '"'))
        quote = '"';
    else
        return false;

    while (!completeQuotes && buffer.hasCharactersRemaining()) {
        skipUntil(buffer, static_cast<CharacterType>(quote));
        if (startSpan[buffer.position() - startSpan.data() - 1] != '\\')
            completeQuotes = true;
        completeQuotes = skipExactly(buffer, static_cast<CharacterType>(quote)) && completeQuotes;
    }
    return true;
}

// Before:
//
// <cat.jpg>; rel=preload; foo=bar
//                ^               ^
//            position            end
//
// After (if successful: otherwise the method returns false)
//
// <cat.jpg>; rel=preload; foo=bar
//                       ^        ^
//                   position     end
template<typename CharacterType> static bool parseParameterValue(StringParsingBuffer<CharacterType>& buffer, String& value)
{
    auto valueStart = buffer.span();
    size_t valueLength = 0;
    bool completeQuotes = false;
    bool hasQuotes = skipQuotesIfNeeded(buffer, completeQuotes);
    if (!hasQuotes)
        skipWhile<isParameterValueChar>(buffer);
    valueLength = buffer.position() - valueStart.data();
    skipWhile<isTabOrSpace>(buffer);
    if ((!completeQuotes && !valueLength) || (!buffer.atEnd() && !isParameterValueEnd(*buffer))) {
        value = emptyString();
        return false;
    }
    if (hasQuotes) {
        skip(valueStart, 1);
        ASSERT(valueLength);
        --valueLength;
    }
    if (completeQuotes) {
        ASSERT(valueLength);
        --valueLength;
    }
    value = String(valueStart.first(valueLength));
    return !hasQuotes || completeQuotes;
}

void LinkHeader::setValue(LinkParameterName name, String&& value)
{
    switch (name) {
    case LinkParameterRel:
        if (!m_rel)
            m_rel = WTFMove(value);
        break;
    case LinkParameterAnchor:
        m_isValid = false;
        break;
    case LinkParameterCrossOrigin:
        m_crossOrigin = WTFMove(value);
        break;
    case LinkParameterAs:
        m_as = WTFMove(value);
        break;
    case LinkParameterType:
        m_mimeType = WTFMove(value);
        break;
    case LinkParameterMedia:
        m_media = WTFMove(value);
        break;
    case LinkParameterImageSrcSet:
        m_imageSrcSet = WTFMove(value);
        break;
    case LinkParameterImageSizes:
        m_imageSizes = WTFMove(value);
        break;
    case LinkParameterNonce:
        m_nonce = WTFMove(value);
        break;
    case LinkParameterReferrerPolicy:
        m_referrerPolicy = WTFMove(value);
        break;
    case LinkParameterFetchPriority:
        m_fetchPriority = WTFMove(value);
        break;
    case LinkParameterTitle:
    case LinkParameterRev:
    case LinkParameterHreflang:
    case LinkParameterUnknown:
        // These parameters are not yet supported, so they are currently ignored.
        break;
    }
    // FIXME: Add support for more header parameters as neccessary.
}

template<typename CharacterType> static void findNextHeader(StringParsingBuffer<CharacterType>& buffer)
{
    skipUntil(buffer, ',');
    skipExactly(buffer, ',');
}

template<typename CharacterType> LinkHeader::LinkHeader(StringParsingBuffer<CharacterType>& buffer)
{
    auto urlResult = findURLBoundaries(buffer);
    if (urlResult == std::nullopt) {
        m_isValid = false;
        findNextHeader(buffer);
        return;
    }
    m_url = urlResult.value();

    while (m_isValid && buffer.hasCharactersRemaining()) {
        if (!parseParameterDelimiter(buffer, m_isValid)) {
            findNextHeader(buffer);
            return;
        }

        auto parameterName = parseParameterName(buffer);
        if (!parameterName) {
            findNextHeader(buffer);
            m_isValid = false;
            return;
        }

        String parameterValue;
        if (!parseParameterValue(buffer, parameterValue) && !isExtensionParameter(*parameterName)) {
            findNextHeader(buffer);
            m_isValid = false;
            return;
        }

        setValue(*parameterName, WTFMove(parameterValue));
    }
    findNextHeader(buffer);
}

LinkHeaderSet::LinkHeaderSet(const String& header)
{
    readCharactersForParsing(header, [&](auto buffer) {
        while (buffer.hasCharactersRemaining())
            m_headerSet.append(LinkHeader { buffer });
    });
}

} // namespace WebCore
