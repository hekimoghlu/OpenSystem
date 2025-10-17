/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "SpellChecker.h"

#include "Document.h"
#include "DocumentMarkerController.h"
#include "Editing.h"
#include "Editor.h"
#include "EditorClient.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PositionIterator.h"
#include "Range.h"
#include "RenderObject.h"
#include "Settings.h"
#include "TextCheckerClient.h"
#include "TextIterator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SpellChecker);

SpellCheckRequest::SpellCheckRequest(const SimpleRange& checkingRange, const SimpleRange& automaticReplacementRange, const SimpleRange& paragraphRange, const String& text, OptionSet<TextCheckingType> options, TextCheckingProcessType type)
    : m_checkingRange(checkingRange)
    , m_automaticReplacementRange(automaticReplacementRange)
    , m_paragraphRange(paragraphRange)
    , m_rootEditableElement(m_checkingRange.start.container->rootEditableElement())
    , m_requestData(std::nullopt, text, options, type)
{
}

SpellCheckRequest::~SpellCheckRequest() = default;

RefPtr<SpellCheckRequest> SpellCheckRequest::create(OptionSet<TextCheckingType> options, TextCheckingProcessType type, const SimpleRange& checkingRange, const SimpleRange& automaticReplacementRange, const SimpleRange& paragraphRange)
{
    String text = plainText(checkingRange);
    if (text.isEmpty())
        return nullptr;
    return adoptRef(*new SpellCheckRequest(checkingRange, automaticReplacementRange, paragraphRange, text, options, type));
}

const TextCheckingRequestData& SpellCheckRequest::data() const
{
    return m_requestData;
}

void SpellCheckRequest::didSucceed(const Vector<TextCheckingResult>& results)
{
    if (!m_checker)
        return;

    Ref<SpellCheckRequest> protectedThis(*this);
    m_checker->didCheckSucceed(m_requestData.identifier().value(), results);
    m_checker = nullptr;
}

void SpellCheckRequest::didCancel()
{
    if (!m_checker)
        return;

    Ref<SpellCheckRequest> protectedThis(*this);
    m_checker->didCheckCancel(m_requestData.identifier().value());
    m_checker = nullptr;
}

void SpellCheckRequest::setCheckerAndIdentifier(SpellChecker* requester, TextCheckingRequestIdentifier identifier)
{
    ASSERT(!m_checker);
    ASSERT(!m_requestData.identifier());
    m_checker = requester;
    m_requestData.m_identifier = identifier;
}

void SpellCheckRequest::requesterDestroyed()
{
    m_checker = nullptr;
}

SpellChecker::SpellChecker(Editor& editor)
    : m_editor(editor)
    , m_timerToProcessQueuedRequest(*this, &SpellChecker::timerFiredToProcessQueuedRequest)
{
}

SpellChecker::~SpellChecker()
{
    if (m_processingRequest)
        m_processingRequest->requesterDestroyed();
    for (auto& queue : m_requestQueue)
        queue->requesterDestroyed();
}

void SpellChecker::ref() const
{
    m_editor->ref();
}

void SpellChecker::deref() const
{
    m_editor->deref();
}

TextCheckerClient* SpellChecker::client() const
{
    RefPtr page = document().page();
    if (!page)
        return nullptr;
    return page->editorClient().textChecker();
}

void SpellChecker::timerFiredToProcessQueuedRequest()
{
    ASSERT(!m_requestQueue.isEmpty());
    if (m_requestQueue.isEmpty())
        return;

    invokeRequest(m_requestQueue.takeFirst());
}

bool SpellChecker::isAsynchronousEnabled() const
{
    return document().settings().asynchronousSpellCheckingEnabled();
}

bool SpellChecker::canCheckAsynchronously(const SimpleRange& range) const
{
    return client() && isCheckable(range) && isAsynchronousEnabled();
}

bool SpellChecker::isCheckable(const SimpleRange& range) const
{
    bool foundRenderer = false;
    for (Ref node : intersectingNodes(range)) {
        if (node->renderer()) {
            foundRenderer = true;
            break;
        }
    }
    if (!foundRenderer)
        return false;
    RefPtr element = dynamicDowncast<Element>(range.start.container.get());
    return !element || element->isSpellCheckingEnabled();
}

void SpellChecker::requestCheckingFor(Ref<SpellCheckRequest>&& request)
{
    if (!canCheckAsynchronously(request->paragraphRange()))
        return;

    ASSERT(!request->data().identifier());
    auto identifier = TextCheckingRequestIdentifier::generate();

    m_lastRequestIdentifier = identifier;
    request->setCheckerAndIdentifier(this, identifier);

    if (m_timerToProcessQueuedRequest.isActive() || m_processingRequest) {
        enqueueRequest(WTFMove(request));
        return;
    }

    invokeRequest(WTFMove(request));
}

void SpellChecker::invokeRequest(Ref<SpellCheckRequest>&& request)
{
    ASSERT(!m_processingRequest);
    if (!client())
        return;
    m_processingRequest = WTFMove(request);
    client()->requestCheckingOfString(*m_processingRequest, protectedDocument()->selection().selection());
}

void SpellChecker::enqueueRequest(Ref<SpellCheckRequest>&& request)
{
    for (auto& queue : m_requestQueue) {
        if (request->rootEditableElement() != queue->rootEditableElement())
            continue;

        queue = WTFMove(request);
        return;
    }

    m_requestQueue.append(WTFMove(request));
}

void SpellChecker::didCheck(TextCheckingRequestIdentifier identifier, const Vector<TextCheckingResult>& results)
{
    ASSERT(m_processingRequest);
    ASSERT(m_processingRequest->data().identifier() == identifier);
    if (m_processingRequest->data().identifier() != identifier) {
        m_requestQueue.clear();
        return;
    }

    protectedDocument()->editor().markAndReplaceFor(*m_processingRequest, results);

    if (!m_lastProcessedIdentifier || *m_lastProcessedIdentifier < identifier)
        m_lastProcessedIdentifier = identifier;

    m_processingRequest = nullptr;
    if (!m_requestQueue.isEmpty())
        m_timerToProcessQueuedRequest.startOneShot(0_s);
}

Document& SpellChecker::document() const
{
    return m_editor->document();
}

Ref<Document> SpellChecker::protectedDocument() const
{
    return m_editor->document();
}

void SpellChecker::didCheckSucceed(TextCheckingRequestIdentifier identifier, const Vector<TextCheckingResult>& results)
{
    TextCheckingRequestData requestData = m_processingRequest->data();
    if (requestData.identifier() == identifier) {
        OptionSet<DocumentMarkerType> markerTypes;
        if (requestData.checkingTypes().contains(TextCheckingType::Spelling))
            markerTypes.add(DocumentMarkerType::Spelling);
        if (requestData.checkingTypes().contains(TextCheckingType::Grammar))
            markerTypes.add(DocumentMarkerType::Grammar);
        if (!markerTypes.isEmpty())
            removeMarkers(m_processingRequest->checkingRange(), markerTypes);
    }
    didCheck(identifier, results);
}

void SpellChecker::didCheckCancel(TextCheckingRequestIdentifier identifier)
{
    didCheck(identifier, Vector<TextCheckingResult>());
}

} // namespace WebCore
