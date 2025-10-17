/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#include "LocalDOMWindowSpeechSynthesis.h"

#if ENABLE(SPEECH_SYNTHESIS)

#include "LocalDOMWindow.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LocalDOMWindowSpeechSynthesis);

LocalDOMWindowSpeechSynthesis::LocalDOMWindowSpeechSynthesis(DOMWindow* window)
    : LocalDOMWindowProperty(dynamicDowncast<LocalDOMWindow>(window))
{
}

LocalDOMWindowSpeechSynthesis::~LocalDOMWindowSpeechSynthesis() = default;

ASCIILiteral LocalDOMWindowSpeechSynthesis::supplementName()
{
    return "LocalDOMWindowSpeechSynthesis"_s;
}

// static
LocalDOMWindowSpeechSynthesis* LocalDOMWindowSpeechSynthesis::from(DOMWindow* window)
{
    RefPtr localWindow = dynamicDowncast<LocalDOMWindow>(window);
    if (!localWindow)
        return nullptr;
    auto* supplement = static_cast<LocalDOMWindowSpeechSynthesis*>(Supplement<LocalDOMWindow>::from(localWindow.get(), supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<LocalDOMWindowSpeechSynthesis>(window);
        supplement = newSupplement.get();
        provideTo(localWindow.get(), supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

// static
SpeechSynthesis* LocalDOMWindowSpeechSynthesis::speechSynthesis(DOMWindow& window)
{
    return LocalDOMWindowSpeechSynthesis::from(&window)->speechSynthesis();
}

SpeechSynthesis* LocalDOMWindowSpeechSynthesis::speechSynthesis()
{
    if (!m_speechSynthesis && frame() && frame()->document())
        m_speechSynthesis = SpeechSynthesis::create(*frame()->document());
    return m_speechSynthesis.get();
}

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
