/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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

#include "Highlight.h"
#include "HighlightVisibility.h"
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class DOMMapAdapter;
class DOMString;
class Highlight;
class StaticRange;

class HighlightRegistry : public RefCounted<HighlightRegistry> {
public:
    static Ref<HighlightRegistry> create() { return adoptRef(*new HighlightRegistry); }

    void initializeMapLike(DOMMapAdapter&);
    void setFromMapLike(AtomString&&, Ref<Highlight>&&);
    void clear();
    bool remove(const AtomString&);
    bool isEmpty() const { return map().isEmpty(); }    

    HighlightVisibility highlightsVisibility() const { return m_highlightVisibility; }
#if ENABLE(APP_HIGHLIGHTS)
    WEBCORE_EXPORT void setHighlightVisibility(HighlightVisibility);
#endif
    
    WEBCORE_EXPORT void addAnnotationHighlightWithRange(Ref<StaticRange>&&);
    const HashMap<AtomString, Ref<Highlight>>& map() const { return m_map; }
    const Vector<AtomString>& highlightNames() const { return m_highlightNames; }
    
private:
    HighlightRegistry() = default;
    HashMap<AtomString, Ref<Highlight>> m_map;
    Vector<AtomString> m_highlightNames;

    HighlightVisibility m_highlightVisibility { HighlightVisibility::Hidden };
};

}

