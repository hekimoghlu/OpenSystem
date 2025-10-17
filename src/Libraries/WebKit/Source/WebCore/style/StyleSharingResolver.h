/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

#include "Styleable.h"
#include <wtf/HashMap.h>

namespace WebCore {

class Document;
class Element;
class Node;
class RenderStyle;
class SpaceSplitString;
class StyledElement;

namespace Style {

class RuleSet;
class ScopeRuleSets;
class Update;
struct SelectorMatchingState;

class SharingResolver {
public:
    SharingResolver(const Document&, const ScopeRuleSets&, SelectorMatchingState&);

    std::unique_ptr<RenderStyle> resolve(const Styleable&, const Update&);

private:
    struct Context;

    StyledElement* findSibling(const Context&, Node*, unsigned& count) const;
    Node* locateCousinList(const Element* parent) const;
    bool canShareStyleWithElement(const Context&, const StyledElement& candidateElement) const;
    bool styleSharingCandidateMatchesRuleSet(const StyledElement&, const RuleSet*) const;
    bool sharingCandidateHasIdenticalStyleAffectingAttributes(const Context&, const StyledElement& sharingCandidate) const;
    bool classNamesAffectedByRules(const SpaceSplitString& classNames) const;

    CheckedRef<const Document> m_document;
    const ScopeRuleSets& m_ruleSets;
    SelectorMatchingState& m_selectorMatchingState;

    // FIXME: Use WeakHashMap or UncheckedKeyHashMap<CheckedPtr, CheckedPtr>.
    UncheckedKeyHashMap<const Element*, const Element*> m_elementsSharingStyle;
};

}
}
