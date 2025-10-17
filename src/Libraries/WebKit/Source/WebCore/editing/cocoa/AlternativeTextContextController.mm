/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#import "config.h"
#import "AlternativeTextContextController.h"

namespace WebCore {

std::optional<DictationContext> AlternativeTextContextController::addAlternatives(PlatformTextAlternatives *alternatives)
{
    if (!alternatives)
        return { };
    return m_contexts.ensure(alternatives, [&] {
        auto context = DictationContext::generate();
        m_alternatives.add(context, alternatives);
        return context;
    }).iterator->value;
}

void AlternativeTextContextController::replaceAlternatives(PlatformTextAlternatives *alternatives, DictationContext context)
{
    removeAlternativesForContext(context);
    if (!alternatives)
        return;

    m_contexts.set(alternatives, context);
    m_alternatives.set(context, alternatives);
}

PlatformTextAlternatives *AlternativeTextContextController::alternativesForContext(DictationContext context) const
{
    return m_alternatives.get(context).get();
}

void AlternativeTextContextController::removeAlternativesForContext(DictationContext context)
{
    if (auto alternatives = m_alternatives.take(context))
        m_contexts.remove(alternatives);
}

void AlternativeTextContextController::clear()
{
    m_alternatives.clear();
    m_contexts.clear();
}

} // namespace WebCore
