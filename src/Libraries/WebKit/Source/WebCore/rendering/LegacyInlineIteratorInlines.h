/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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

#include "LegacyInlineIterator.h"
#include "RenderStyleInlines.h"

namespace WebCore {

inline bool LegacyInlineIterator::atTextParagraphSeparator() const
{
    auto* textRenderer = dynamicDowncast<RenderText>(m_renderer);
    return textRenderer
        && m_renderer->preservesNewline()
        && textRenderer->characterAt(m_pos) == '\n';
}

inline bool LegacyInlineIterator::atParagraphSeparator() const
{
    return (m_renderer && m_renderer->isBR()) || atTextParagraphSeparator();
}

inline void IsolateTracker::addFakeRunIfNecessary(RenderObject& object, unsigned position, unsigned end, RenderElement& root, InlineBidiResolver& resolver)
{
    // We only need to add a fake run for a given isolated span once during each call to createBidiRunsForLine.
    // We'll be called for every span inside the isolated span so we just ignore subsequent calls.
    // We also avoid creating a fake run until we hit a child that warrants one, e.g. we skip floats.
    if (RenderBlock::shouldSkipCreatingRunsForObject(object))
        return;
    if (!m_haveAddedFakeRunForRootIsolate) {
        // object and position together denote a single position in the inline, from which the parsing of the isolate will start.
        // We don't need to mark the end of the run because this is implicit: it is either endOfLine or the end of the
        // isolate, when we call createBidiRunsForLine it will stop at whichever comes first.
        addPlaceholderRunForIsolatedInline(resolver, object, position, root);
    }
    m_haveAddedFakeRunForRootIsolate = true;
    LegacyLineLayout::appendRunsForObject(nullptr, position, end, object, resolver);
}

template<> inline void InlineBidiResolver::appendRunInternal()
{
    if (!m_emptyRun && !m_eor.atEnd() && !m_reachedEndOfLine) {
        // Keep track of when we enter/leave "unicode-bidi: isolate" inlines.
        // Initialize our state depending on if we're starting in the middle of such an inline.
        // FIXME: Could this initialize from this->inIsolate() instead of walking up the render tree?
        IsolateTracker isolateTracker(numberOfIsolateAncestors(m_sor));
        int start = m_sor.offset();
        RenderObject* obj = m_sor.renderer();
        while (obj && obj != m_eor.renderer() && obj != endOfLine.renderer()) {
            if (isolateTracker.inIsolate())
                isolateTracker.addFakeRunIfNecessary(*obj, start, obj->length(), *m_sor.root(), *this);
            else
                LegacyLineLayout::appendRunsForObject(&m_runs, start, obj->length(), *obj, *this);
            // FIXME: start/obj should be an LegacyInlineIterator instead of two separate variables.
            start = 0;
            obj = nextInlineRendererSkippingEmpty(*m_sor.root(), obj, &isolateTracker);
        }
        if (obj) {
            unsigned pos = obj == m_eor.renderer() ? m_eor.offset() : UINT_MAX;
            if (obj == endOfLine.renderer() && endOfLine.offset() <= pos) {
                m_reachedEndOfLine = true;
                pos = endOfLine.offset();
            }
            // It's OK to add runs for zero-length RenderObjects, just don't make the run larger than it should be
            int end = obj->length() ? pos + 1 : 0;
            if (isolateTracker.inIsolate())
                isolateTracker.addFakeRunIfNecessary(*obj, start, obj->length(), *m_sor.root(), *this);
            else
                LegacyLineLayout::appendRunsForObject(&m_runs, start, end, *obj, *this);
        }

        m_eor.increment();
        m_sor = m_eor;
    }

    m_direction = U_OTHER_NEUTRAL;
    m_status.eor = U_OTHER_NEUTRAL;
}

} // namespace WebCore
