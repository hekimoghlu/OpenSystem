/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#import <WebKitLegacy/WebNSDataExtras.h>

#import <wtf/Assertions.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/SpanCocoa.h>
#import <wtf/text/ParsingUtilities.h>
#import <wtf/text/StringCommon.h>

@implementation NSData (WebNSDataExtras)

- (NSString *)_webkit_guessedMIMETypeForXML
{
    auto bytes = span(self);

    constexpr size_t channelTagLength = 7;

    size_t remaining = std::min<size_t>(bytes.size(), WEB_GUESS_MIME_TYPE_PEEK_LENGTH) - (channelTagLength - 1);
    bytes = bytes.first(remaining);

    BOOL foundRDF = false;

    while (!bytes.empty()) {
        // Look for a "<".
        auto hitIndex = WTF::find(bytes, '<');
        if (hitIndex == notFound)
            break;

        // We are trying to identify RSS or Atom. RSS has a top-level
        // element of either <rss> or <rdf>. However, there are
        // non-RSS RDF files, so in the case of <rdf> we further look
        // for a <channel> element. In the case of an Atom file, a
        // top-level <feed> element is all we need to see. Only tags
        // starting with <? or <! can precede the root element. We
        // bail if we don't find an <rss>, <feed> or <rdf> element
        // right after those.

        auto hit = bytes.subspan(hitIndex);
        if (foundRDF) {
            if (spanHasPrefixIgnoringASCIICase(hit, "<channel"_span))
                return @"application/rss+xml";
        } else if (spanHasPrefixIgnoringASCIICase(hit, "<rdf"_span))
            foundRDF = TRUE;
        else if (spanHasPrefixIgnoringASCIICase(hit, "<rss"_span))
            return @"application/rss+xml";
        else if (spanHasPrefixIgnoringASCIICase(hit, "<feed"_span))
            return @"application/atom+xml";
        else if (!spanHasPrefixIgnoringASCIICase(hit, "<?"_span) && !spanHasPrefixIgnoringASCIICase(hit, "<!"_span))
            return nil;

        // Skip the "<" and continue.
        skip(bytes, hitIndex + 1);
    }

    return nil;
}

- (NSString *)_webkit_guessedMIMEType
{
    constexpr size_t scriptTagLength = 7;
    constexpr size_t textHTMLLength = 9;

    NSString *MIMEType = [self _webkit_guessedMIMETypeForXML];
    if ([MIMEType length])
        return MIMEType;

    auto bytes = span(self);

    size_t remaining = std::min<size_t>(bytes.size(), WEB_GUESS_MIME_TYPE_PEEK_LENGTH) - (scriptTagLength - 1);
    auto cursor = bytes.first(remaining);
    while (!cursor.empty()) {
        // Look for a "<".
        size_t hitIndex = WTF::find(cursor, '<');
        if (hitIndex == notFound)
            break;

        auto hit = cursor.subspan(hitIndex);
        // If we found a "<", look for "<html>" or "<a " or "<script".
        if (spanHasPrefixIgnoringASCIICase(hit, "<html>"_span)
            || spanHasPrefixIgnoringASCIICase(hit, "<a "_span)
            || spanHasPrefixIgnoringASCIICase(hit, "<script"_span)
            || spanHasPrefixIgnoringASCIICase(hit, "<title>"_span)) {
            return @"text/html";
        }

        // Skip the "<" and continue.
        skip(cursor, hitIndex + 1);
    }

    // Test for a broken server which has sent the content type as part of the content.
    // This code could be improved to look for other mime types.
    remaining = std::min<size_t>(bytes.size(), WEB_GUESS_MIME_TYPE_PEEK_LENGTH) - (textHTMLLength - 1);
    cursor = bytes.first(remaining);
    while (!cursor.empty()) {
        // Look for a "t" or "T".
        size_t lowerHitIndex = WTF::find(cursor, 't');
        size_t upperHitIndex = WTF::find(cursor, 'T');
        if (lowerHitIndex == notFound && upperHitIndex == notFound)
            break;

        static_assert(notFound == std::numeric_limits<size_t>::max());
        size_t hitIndex = std::min(lowerHitIndex, upperHitIndex);
        auto hit = cursor.subspan(hitIndex);

        // If we found a "t/T", look for "text/html".
        if (spanHasPrefixIgnoringASCIICase(hit, "text/html"_span))
            return @"text/html";

        // Skip the "t/T" and continue.
        skip(cursor, hitIndex + 1);
    }

    if (spanHasPrefix(bytes, "BEGIN:VCARD"_span))
        return @"text/vcard";
    if (spanHasPrefix(bytes, "BEGIN:VCALENDAR"_span))
        return @"text/calendar";

    // Test for plain text.
    bool foundBadCharacter = false;
    for (auto c : bytes) {
        if ((c < 0x20 || c > 0x7E) && (c != '\t' && c != '\r' && c != '\n')) {
            foundBadCharacter = true;
            break;
        }
    }
    if (!foundBadCharacter) {
        // Didn't encounter any bad characters, looks like plain text.
        return @"text/plain";
    }

    // Looks like this is a binary file.

    // Sniff for the JPEG magic number.
    constexpr std::array<uint8_t, 4> jpegMagicNumber { 0xFF, 0xD8, 0xFF, 0xE0 };
    if (spanHasPrefix(bytes, std::span { jpegMagicNumber }))
        return @"image/jpeg";

    return nil;
}

- (BOOL)_web_isCaseInsensitiveEqualToCString:(const char *)string
{
    ASSERT(string);
    return equalLettersIgnoringASCIICase(span(self), unsafeSpan(string));
}

@end
