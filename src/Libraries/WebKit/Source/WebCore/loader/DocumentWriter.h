/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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

#include "NavigationAction.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Document;
class DocumentParser;
class LocalFrame;
class SharedBuffer;
class TextResourceDecoder;

class DocumentWriter {
    WTF_MAKE_NONCOPYABLE(DocumentWriter);
public:
    DocumentWriter() = default;

    void replaceDocumentWithResultOfExecutingJavascriptURL(const String&, Document* ownerDocument);

    bool begin();
    bool begin(const URL&, bool dispatchWindowObjectAvailable = true, Document* ownerDocument = nullptr, std::optional<ScriptExecutionContextIdentifier> = std::nullopt, const NavigationAction* triggeringAction = nullptr);
    void addData(const SharedBuffer&);
    void insertDataSynchronously(const String&); // For an internal use only to prevent the parser from yielding.
    WEBCORE_EXPORT void end();

    void setFrame(LocalFrame&);

    enum class IsEncodingUserChosen : bool { No, Yes };
    WEBCORE_EXPORT void setEncoding(const String& encoding, IsEncodingUserChosen);

    const String& mimeType() const { return m_mimeType; }
    void setMIMEType(const String& type) { m_mimeType = type; }

    // Exposed for DocumentParser::appendBytes.
    TextResourceDecoder& decoder();
    void reportDataReceived();

    void setDocumentWasLoadedAsPartOfNavigation();

private:
    Ref<Document> createDocument(const URL&, std::optional<ScriptExecutionContextIdentifier>);
    void clear();
    RefPtr<DocumentParser> protectedParser() const;

    WeakPtr<LocalFrame> m_frame;

    String m_mimeType;

    String m_encoding;
    RefPtr<TextResourceDecoder> m_decoder;
    RefPtr<DocumentParser> m_parser;

    enum class State : uint8_t { NotStarted, Started, Finished };
    State m_state { State::NotStarted };

    bool m_hasReceivedSomeData { false };
    bool m_encodingWasChosenByUser { false };
};

} // namespace WebCore
