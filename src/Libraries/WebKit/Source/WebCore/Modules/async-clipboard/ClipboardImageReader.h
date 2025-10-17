/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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

#include "Blob.h"
#include "Document.h"
#include "Pasteboard.h"

namespace WebCore {

class Document;
class FragmentedSharedBuffer;

struct ClipboardImageReader : PasteboardFileReader {
    ClipboardImageReader(Document* document, const String& mimeType)
        : PasteboardFileReader()
        , m_document(document)
        , m_mimeType(mimeType)
    {
    }

    RefPtr<Blob> takeResult() { return std::exchange(m_result, nullptr); }

private:
    void readFilename(const String&) final { ASSERT_NOT_REACHED(); }

    bool shouldReadBuffer(const String&) const final;
    void readBuffer(const String& filename, const String& type, Ref<SharedBuffer>&&) final;

    RefPtr<Document> m_document;
    String m_mimeType;
    RefPtr<Blob> m_result;
};

} // namespace WebCore
