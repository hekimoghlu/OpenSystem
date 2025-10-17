/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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

#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class CSSStyleRule;
class CSSStyleSheet;
class ExtensionStyleSheets;
class StyleRule;
class StyleSheetContents;

namespace Style {

class Scope;

class InspectorCSSOMWrappers {
public:
    // WARNING. This will construct CSSOM wrappers for all style rules and cache them in a map for significant memory cost.
    // It is here to support inspector. Don't use for any regular engine functions.
    CSSStyleRule* getWrapperForRuleInSheets(const StyleRule*);
    void collectFromStyleSheetIfNeeded(CSSStyleSheet*);
    void collectDocumentWrappers(ExtensionStyleSheets&);
    void collectScopeWrappers(Scope&);

private:
    template <class ListType>
    void collect(ListType*);

    void collectFromStyleSheetContents(StyleSheetContents*);
    void collectFromStyleSheets(const Vector<RefPtr<CSSStyleSheet>>&);
    void maybeCollectFromStyleSheets(const Vector<RefPtr<CSSStyleSheet>>&);

    UncheckedKeyHashMap<const StyleRule*, RefPtr<CSSStyleRule>> m_styleRuleToCSSOMWrapperMap;
    UncheckedKeyHashSet<RefPtr<CSSStyleSheet>> m_styleSheetCSSOMWrapperSet;
};

} // namespace Style
} // namespace WebCore
