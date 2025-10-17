/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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
#include "BidiRun.h"
#include "LegacyInlineBox.h"
#include <wtf/RefCountedLeakCounter.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BidiRun);

DEFINE_DEBUG_ONLY_GLOBAL(WTF::RefCountedLeakCounter, bidiRunCounter, ("BidiRun"));

BidiRun::BidiRun(unsigned start, unsigned stop, RenderObject& renderer, BidiContext* context, UCharDirection dir)
    : BidiCharacterRun(start, stop, context, dir)
    , m_renderer(renderer)
    , m_box(nullptr)
{
#ifndef NDEBUG
    bidiRunCounter.increment();
#endif
    ASSERT(!is<RenderText>(m_renderer) || static_cast<unsigned>(stop) <= downcast<RenderText>(m_renderer).text().length());
}

BidiRun::~BidiRun()
{
#ifndef NDEBUG
    bidiRunCounter.decrement();
#endif
}

std::unique_ptr<BidiRun> BidiRun::takeNext()
{
    std::unique_ptr<BidiCharacterRun> next = BidiCharacterRun::takeNext();
    BidiCharacterRun* raw = next.release();
    std::unique_ptr<BidiRun> result = std::unique_ptr<BidiRun>(static_cast<BidiRun*>(raw));
    return result;
}

}
