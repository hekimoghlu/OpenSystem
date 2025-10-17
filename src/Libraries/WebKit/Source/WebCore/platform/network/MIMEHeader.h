/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

#if ENABLE(MHTML)

#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class SharedBufferChunkReader;

// FIXME: This class is a limited MIME parser used to parse the MIME headers of MHTML files.
class MIMEHeader : public RefCounted<MIMEHeader> {
public:
    enum Encoding {
        QuotedPrintable,
        Base64,
        SevenBit,
        Binary,
        Unknown
    };

    static RefPtr<MIMEHeader> parseHeader(SharedBufferChunkReader& crLFLineReader);

    bool isMultipart() const { return m_contentType.startsWith("multipart/"_s); }

    String contentType() const { return m_contentType; }
    String charset() const { return m_charset; }
    Encoding contentTransferEncoding() const { return m_contentTransferEncoding; }
    String contentLocation() const { return m_contentLocation; }

    // Multi-part type and boundaries are only valid for multipart MIME headers.
    String multiPartType() const { return m_multipartType; }
    String endOfPartBoundary() const { return m_endOfPartBoundary; }
    String endOfDocumentBoundary() const { return m_endOfDocumentBoundary; }

private:
    MIMEHeader();

    static Encoding parseContentTransferEncoding(StringView);

    String m_contentType;
    String m_charset;
    Encoding m_contentTransferEncoding;
    String m_contentLocation;
    String m_multipartType;
    String m_endOfPartBoundary;
    String m_endOfDocumentBoundary;
};

}

#endif
