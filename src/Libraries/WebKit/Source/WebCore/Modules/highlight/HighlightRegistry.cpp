/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include "HighlightRegistry.h"

#include "IDLTypes.h"
#include "JSDOMMapLike.h"
#include "JSHighlight.h"

namespace WebCore {
    
void HighlightRegistry::initializeMapLike(DOMMapAdapter& map)
{
    for (auto& keyValue : m_map)
        map.set<IDLDOMString, IDLInterface<Highlight>>(keyValue.key, keyValue.value);
}

void HighlightRegistry::setFromMapLike(AtomString&& key, Ref<Highlight>&& value)
{
    auto addResult = m_map.set(key, WTFMove(value));
    if (addResult.isNewEntry) {
        ASSERT(!m_highlightNames.contains(key));
        m_highlightNames.append(WTFMove(key));
    }
}

void HighlightRegistry::clear()
{
    m_highlightNames.clear();
    auto map = std::exchange(m_map, { });
    for (auto& highlight : map.values())
        highlight->clearFromSetLike();
}

bool HighlightRegistry::remove(const AtomString& key)
{
    m_highlightNames.removeFirst(key);
    return m_map.remove(key);
}
#if ENABLE(APP_HIGHLIGHTS)
void HighlightRegistry::setHighlightVisibility(HighlightVisibility highlightVisibility)
{
    if (m_highlightVisibility == highlightVisibility)
        return;
    
    m_highlightVisibility = highlightVisibility;
    
    for (auto& highlight : m_map)
        highlight.value->repaint();
}
#endif
static ASCIILiteral annotationHighlightKey()
{
    return "annotationHighlightKey"_s;
}

void HighlightRegistry::addAnnotationHighlightWithRange(Ref<StaticRange>&& value)
{
    if (m_map.contains(annotationHighlightKey()))
        m_map.get(annotationHighlightKey())->addToSetLike(value);
    else
        setFromMapLike(annotationHighlightKey(), Highlight::create({ std::ref<AbstractRange>(value.get()) }));
}

}
