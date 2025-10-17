/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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

#include "Element.h"
#include "SimpleRange.h"
#include "TextChecking.h"
#include "Timer.h"
#include <wtf/Deque.h>
#include <wtf/Markable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Editor;
class SpellChecker;
class TextCheckerClient;

class SpellCheckRequest final : public TextCheckingRequest {
public:
    static RefPtr<SpellCheckRequest> create(OptionSet<TextCheckingType>, TextCheckingProcessType, const SimpleRange& checkingRange, const SimpleRange& automaticReplacementRange, const SimpleRange& paragraphRange);
    virtual ~SpellCheckRequest();

    const SimpleRange& checkingRange() const { return m_checkingRange; }
    const SimpleRange& paragraphRange() const { return m_paragraphRange; }
    const SimpleRange& automaticReplacementRange() const { return m_automaticReplacementRange; }
    Element* rootEditableElement() const { return m_rootEditableElement.get(); }

    void setCheckerAndIdentifier(SpellChecker*, TextCheckingRequestIdentifier);
    void requesterDestroyed();

    const TextCheckingRequestData& data() const final;

private:
    void didSucceed(const Vector<TextCheckingResult>&) final;
    void didCancel() final;

    SpellCheckRequest(const SimpleRange& checkingRange, const SimpleRange& automaticReplacementRange, const SimpleRange& paragraphRange, const String&, OptionSet<TextCheckingType>, TextCheckingProcessType);

    SingleThreadWeakPtr<SpellChecker> m_checker;
    SimpleRange m_checkingRange;
    SimpleRange m_automaticReplacementRange;
    SimpleRange m_paragraphRange;
    RefPtr<Element> m_rootEditableElement;
    TextCheckingRequestData m_requestData;
};

class SpellChecker : public CanMakeSingleThreadWeakPtr<SpellChecker> {
    WTF_MAKE_TZONE_ALLOCATED(SpellChecker);
public:
    friend class SpellCheckRequest;

    explicit SpellChecker(Editor&);
    ~SpellChecker();

    void ref() const;
    void deref() const;

    bool isAsynchronousEnabled() const;
    bool isCheckable(const SimpleRange&) const;

    void requestCheckingFor(Ref<SpellCheckRequest>&&);

    std::optional<TextCheckingRequestIdentifier> lastRequestIdentifier() const { return m_lastRequestIdentifier; }
    std::optional<TextCheckingRequestIdentifier> lastProcessedIdentifier() const { return m_lastProcessedIdentifier; }

private:
    bool canCheckAsynchronously(const SimpleRange&) const;
    TextCheckerClient* client() const;
    void timerFiredToProcessQueuedRequest();
    void invokeRequest(Ref<SpellCheckRequest>&&);
    void enqueueRequest(Ref<SpellCheckRequest>&&);
    void didCheckSucceed(TextCheckingRequestIdentifier, const Vector<TextCheckingResult>&);
    void didCheckCancel(TextCheckingRequestIdentifier);
    void didCheck(TextCheckingRequestIdentifier, const Vector<TextCheckingResult>&);

    Document& document() const;
    Ref<Document> protectedDocument() const;

    WeakRef<Editor> m_editor;
    Markable<TextCheckingRequestIdentifier> m_lastRequestIdentifier;
    Markable<TextCheckingRequestIdentifier> m_lastProcessedIdentifier;

    Timer m_timerToProcessQueuedRequest;

    RefPtr<SpellCheckRequest> m_processingRequest;
    Deque<Ref<SpellCheckRequest>> m_requestQueue;
};

} // namespace WebCore
