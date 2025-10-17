/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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

#include "Document.h"
#include "DocumentParser.h"

namespace WebCore {

class RawDataDocumentParser : public DocumentParser {
protected:
    explicit RawDataDocumentParser(Document& document)
        : DocumentParser(document)
    {
    }

    void finish() override
    {
        if (!isStopped())
            protectedDocument()->finishedParsing();
    }

private:
    void flush(DocumentWriter& writer) override
    {
        // Make sure appendBytes is called at least once.
        appendBytes(writer, { });
    }

    void insert(SegmentedString&&) override
    {
        // <https://bugs.webkit.org/show_bug.cgi?id=25397>: JS code can always call document.write, we need to handle it.
        ASSERT_NOT_REACHED();
    }

    void append(RefPtr<StringImpl>&&) override
    {
        ASSERT_NOT_REACHED();
    }
};

} // namespace WebCore
