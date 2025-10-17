/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#include "DocumentParser.h"

#include "Document.h"
#include "EventTarget.h"
#include <wtf/Assertions.h>

namespace WebCore {

DocumentParser::DocumentParser(Document& document)
    : m_state(ParserState::Parsing)
    , m_documentWasLoadedAsPartOfNavigation(false)
    , m_document(document)
{
}

DocumentParser::~DocumentParser()
{
    // Document is expected to call detach() before releasing its ref.
    // This ASSERT is slightly awkward for parsers with a fragment case
    // as there is no Document to release the ref.
    ASSERT(!m_document);
}

void DocumentParser::startParsing()
{
    m_state = ParserState::Parsing;
}

void DocumentParser::prepareToStopParsing()
{
    ASSERT(m_state == ParserState::Parsing);
    m_state = ParserState::Stopping;
}

void DocumentParser::stopParsing()
{
    m_state = ParserState::Stopped;
}

void DocumentParser::detach()
{
    m_state = ParserState::Detached;
    m_document = nullptr;
}

void DocumentParser::suspendScheduledTasks()
{
}

void DocumentParser::resumeScheduledTasks()
{
}

RefPtr<Document> DocumentParser::protectedDocument() const
{
    return document();
}

} // namespace WebCore
