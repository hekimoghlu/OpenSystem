/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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

#include "DocumentParser.h"

namespace WebCore {

class DecodedDataDocumentParser : public DocumentParser {
public:
    // Only used by the XMLDocumentParser to communicate back to
    // XMLHttpRequest if the responseXML was well formed.
    virtual bool wellFormed() const { return true; }

protected:
    explicit DecodedDataDocumentParser(Document&);

private:
    // append is used by DocumentWriter::replaceDocument.
    void append(RefPtr<StringImpl>&&) override = 0;

    // appendBytes and flush are used by DocumentWriter (the loader).
    void appendBytes(DocumentWriter&, std::span<const uint8_t>) override;
    void flush(DocumentWriter&) override;
};

} // namespace WebCore
