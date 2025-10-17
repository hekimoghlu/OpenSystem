/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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

#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class DocumentWriter;
class HTMLDocumentParser;
class SegmentedString;
class ScriptableDocumentParser;
class WeakPtrImplWithEventTargetData;

class DocumentParser : public RefCounted<DocumentParser> {
public:
    virtual ~DocumentParser();

    virtual ScriptableDocumentParser* asScriptableDocumentParser() { return nullptr; }
    virtual HTMLDocumentParser* asHTMLDocumentParser() { return nullptr; }

    // http://www.whatwg.org/specs/web-apps/current-work/#insertion-point
    virtual bool hasInsertionPoint() { return true; }

    // insert is used by document.write.
    virtual void insert(SegmentedString&&) = 0;

    // appendBytes and flush are used by DocumentWriter (the loader).
    virtual void appendBytes(DocumentWriter&, std::span<const uint8_t>) = 0;
    virtual void flush(DocumentWriter&) = 0;

    virtual void append(RefPtr<StringImpl>&&) = 0;
    virtual void appendSynchronously(RefPtr<StringImpl>&& inputSource) { append(WTFMove(inputSource)); }

    virtual void finish() = 0;

    // FIXME: processingData() is only used by DocumentLoader::isLoadingInAPISense
    // and is very unclear as to what it actually means.  The LegacyHTMLDocumentParser
    // used to implement it.
    virtual bool processingData() const { return false; }

    // document() will return 0 after detach() is called.
    Document* document() const { ASSERT(m_document); return m_document.get(); }
    RefPtr<Document> protectedDocument() const;

    bool isParsing() const { return m_state == ParserState::Parsing; }
    bool isStopping() const { return m_state == ParserState::Stopping; }
    bool isStopped() const { return m_state >= ParserState::Stopped; }
    bool isDetached() const { return m_state == ParserState::Detached; }

    // FIXME: Is this necessary? Does XMLDocumentParserLibxml2 really need to set this?
    virtual void startParsing();

    // prepareToStop() is used when the EOF token is encountered and parsing is to be
    // stopped normally.
    virtual void prepareToStopParsing();

    // stopParsing() is used when a load is canceled/stopped.
    // stopParsing() is currently different from detach(), but shouldn't be.
    // It should NOT be ok to call any methods on DocumentParser after either
    // detach() or stopParsing() but right now only detach() will ASSERT.
    virtual void stopParsing();

    // Document is expected to detach the parser before releasing its ref.
    // After detach, m_document is cleared.  The parser will unwind its
    // callstacks, but not produce any more nodes.
    // It is impossible for the parser to touch the rest of WebCore after
    // detach is called.
    virtual void detach();

    void setDocumentWasLoadedAsPartOfNavigation() { m_documentWasLoadedAsPartOfNavigation = true; }
    bool documentWasLoadedAsPartOfNavigation() const { return m_documentWasLoadedAsPartOfNavigation; }

    // FIXME: The names are not very accurate :(
    virtual void suspendScheduledTasks();
    virtual void resumeScheduledTasks();

    virtual void didBeginYieldingParser() { }
    virtual void didEndYieldingParser() { }

protected:
    explicit DocumentParser(Document&);

private:
    enum class ParserState : uint8_t {
        Parsing,
        Stopping,
        Stopped,
        Detached
    };
    ParserState m_state;
    bool m_documentWasLoadedAsPartOfNavigation;

    // Every DocumentParser needs a pointer back to the document.
    // m_document will be nullptr after the parser is stopped.
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
};

} // namespace WebCore
