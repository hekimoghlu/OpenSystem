/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

#if ENABLE(MHTML)
#include "MHTMLParser.h"

#include "CommonAtomStrings.h"
#include "MHTMLArchive.h"
#include "MIMEHeader.h"
#include "MIMETypeRegistry.h"
#include "QuotedPrintable.h"
#include <wtf/text/Base64.h>

namespace WebCore {

static bool skipLinesUntilBoundaryFound(SharedBufferChunkReader& lineReader, const String& boundary)
{
    String line;
    while (!(line = lineReader.nextChunkAsUTF8StringWithLatin1Fallback()).isNull()) {
        if (line == boundary)
            return true;
    }
    return false;
}

MHTMLParser::MHTMLParser(FragmentedSharedBuffer* data)
    : m_lineReader(data, "\r\n")
{
}

RefPtr<MHTMLArchive> MHTMLParser::parseArchive()
{
    return parseArchiveWithHeader(MIMEHeader::parseHeader(m_lineReader).get());
}

RefPtr<MHTMLArchive> MHTMLParser::parseArchiveWithHeader(MIMEHeader* header)
{
    if (!header) {
        LOG_ERROR("Failed to parse MHTML part: no header.");
        return nullptr;
    }

    auto archive = MHTMLArchive::create();
    if (!header->isMultipart()) {
        // With IE a page with no resource is not multi-part.
        bool endOfArchiveReached = false;
        RefPtr<ArchiveResource> resource = parseNextPart(*header, String(), String(), endOfArchiveReached);
        if (!resource)
            return nullptr;
        archive->setMainResource(resource.releaseNonNull());
        return archive;
    }

    // Skip the message content (it's a generic browser specific message).
    skipLinesUntilBoundaryFound(m_lineReader, header->endOfPartBoundary());

    bool endOfArchive = false;
    while (!endOfArchive) {
        RefPtr<MIMEHeader> resourceHeader = MIMEHeader::parseHeader(m_lineReader);
        if (!resourceHeader) {
            LOG_ERROR("Failed to parse MHTML, invalid MIME header.");
            return nullptr;
        }
        if (resourceHeader->contentType() == "multipart/alternative"_s) {
            // Ignore IE nesting which makes little sense (IE seems to nest only some of the frames).
            RefPtr<MHTMLArchive> subframeArchive = parseArchiveWithHeader(resourceHeader.get());
            if (!subframeArchive) {
                LOG_ERROR("Failed to parse MHTML subframe.");
                return nullptr;
            }
            bool endOfPartReached = skipLinesUntilBoundaryFound(m_lineReader, header->endOfPartBoundary());
            ASSERT_UNUSED(endOfPartReached, endOfPartReached);
            // The top-frame is the first frame found, regardless of the nesting level.
            if (subframeArchive->mainResource())
                addResourceToArchive(subframeArchive->mainResource(), archive.ptr());
            archive->addSubframeArchive(subframeArchive.releaseNonNull());
            continue;
        }

        RefPtr<ArchiveResource> resource = parseNextPart(*resourceHeader, header->endOfPartBoundary(), header->endOfDocumentBoundary(), endOfArchive);
        if (!resource) {
            LOG_ERROR("Failed to parse MHTML part.");
            return nullptr;
        }
        addResourceToArchive(resource.get(), archive.ptr());
    }

    return archive;
}

void MHTMLParser::addResourceToArchive(ArchiveResource* resource, MHTMLArchive* archive)
{
    const String& mimeType = resource->mimeType();
    if (!MIMETypeRegistry::isSupportedNonImageMIMEType(mimeType) || MIMETypeRegistry::isSupportedJavaScriptMIMEType(mimeType) || mimeType == cssContentTypeAtom()) {
        m_resources.append(resource);
        return;
    }

    // The first document suitable resource is the main frame.
    if (!archive->mainResource()) {
        archive->setMainResource(*resource);
        m_frames.append(archive);
        return;
    }

    auto subframe = MHTMLArchive::create();
    subframe->setMainResource(*resource);
    m_frames.append(WTFMove(subframe));
}

RefPtr<ArchiveResource> MHTMLParser::parseNextPart(const MIMEHeader& mimeHeader, const String& endOfPartBoundary, const String& endOfDocumentBoundary, bool& endOfArchiveReached)
{
    ASSERT(endOfPartBoundary.isEmpty() == endOfDocumentBoundary.isEmpty());

    SharedBufferBuilder content;
    const bool checkBoundary = !endOfPartBoundary.isEmpty();
    bool endOfPartReached = false;
    if (mimeHeader.contentTransferEncoding() == MIMEHeader::Binary) {
        if (!checkBoundary) {
            LOG_ERROR("Binary contents requires end of part");
            return nullptr;
        }
        m_lineReader.setSeparator(endOfPartBoundary.utf8().data());
        Vector<uint8_t> part;
        if (!m_lineReader.nextChunk(part)) {
            LOG_ERROR("Binary contents requires end of part");
            return nullptr;
        }
        content.append(WTFMove(part));
        m_lineReader.setSeparator("\r\n");
        Vector<uint8_t> nextChars;
        if (m_lineReader.peek(nextChars, 2) != 2) {
            LOG_ERROR("Invalid seperator.");
            return nullptr;
        }
        endOfPartReached = true;
        ASSERT(nextChars.size() == 2);
        endOfArchiveReached = (nextChars[0] == '-' && nextChars[1] == '-');
        if (!endOfArchiveReached) {
            String line = m_lineReader.nextChunkAsUTF8StringWithLatin1Fallback();
            if (!line.isEmpty()) {
                LOG_ERROR("No CRLF at end of binary section.");
                return nullptr;
            }
        }
    } else {
        String line;
        while (!(line = m_lineReader.nextChunkAsUTF8StringWithLatin1Fallback()).isNull()) {
            endOfArchiveReached = (line == endOfDocumentBoundary);
            if (checkBoundary && (line == endOfPartBoundary || endOfArchiveReached)) {
                endOfPartReached = true;
                break;
            }
            // Note that we use line.utf8() and not line.ascii() as ascii turns special characters (such as tab, line-feed...) into '?'.
            content.append(line.utf8().span());
            if (mimeHeader.contentTransferEncoding() == MIMEHeader::QuotedPrintable) {
                // The line reader removes the \r\n, but we need them for the content in this case as the QuotedPrintable decoder expects CR-LF terminated lines.
                content.append("\r\n"_span);
            }
        }
    }
    if (!endOfPartReached && checkBoundary) {
        LOG_ERROR("No bounday found for MHTML part.");
        return nullptr;
    }

    Vector<uint8_t> data;
    auto contiguousContent = content.takeAsContiguous();
    switch (mimeHeader.contentTransferEncoding()) {
    case MIMEHeader::Base64: {
        auto decodedData = base64Decode(contiguousContent->span());
        if (!decodedData) {
            LOG_ERROR("Invalid base64 content for MHTML part.");
            return nullptr;
        }
        data = WTFMove(*decodedData);
        break;
    }
    case MIMEHeader::QuotedPrintable:
        data = quotedPrintableDecode(contiguousContent->span());
        break;
    case MIMEHeader::SevenBit:
    case MIMEHeader::Binary:
        data.append(contiguousContent->span());
        break;
    default:
        LOG_ERROR("Invalid encoding for MHTML part.");
        return nullptr;
    }
    auto contentBuffer = SharedBuffer::create(WTFMove(data));
    // FIXME: the URL in the MIME header could be relative, we should resolve it if it is.
    // The specs mentions 5 ways to resolve a URL: http://tools.ietf.org/html/rfc2557#section-5
    // IE and Firefox (UNMht) seem to generate only absolute URLs.
    URL location { mimeHeader.contentLocation() };
    return ArchiveResource::create(WTFMove(contentBuffer), location, mimeHeader.contentType(), mimeHeader.charset(), String());
}

size_t MHTMLParser::frameCount() const
{
    return m_frames.size();
}

MHTMLArchive* MHTMLParser::frameAt(size_t index) const
{
    return m_frames[index].get();
}

size_t MHTMLParser::subResourceCount() const
{
    return m_resources.size();
}

ArchiveResource* MHTMLParser::subResourceAt(size_t index) const
{
    return m_resources[index].get();
}

}
#endif
