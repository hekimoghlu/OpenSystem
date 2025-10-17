/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#include "DecodedDataDocumentParser.h"

#include "Document.h"
#include "DocumentWriter.h"
#include "SegmentedString.h"
#include "TextResourceDecoder.h"

namespace WebCore {

DecodedDataDocumentParser::DecodedDataDocumentParser(Document& document)
    : DocumentParser(document)
{
}

void DecodedDataDocumentParser::appendBytes(DocumentWriter& writer, std::span<const uint8_t> data)
{
    if (data.empty())
        return;

    String decoded = writer.decoder().decode(data);
    if (decoded.isEmpty())
        return;

    writer.reportDataReceived();
    append(decoded.releaseImpl());
}

void DecodedDataDocumentParser::flush(DocumentWriter& writer)
{
    String remainingData = writer.decoder().flush();
    if (remainingData.isEmpty())
        return;

    writer.reportDataReceived();
    append(remainingData.releaseImpl());
}

};
