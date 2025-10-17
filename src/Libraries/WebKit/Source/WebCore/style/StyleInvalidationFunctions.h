/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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

#include "HTMLSlotElement.h"
#include "ShadowRoot.h"
#include "StyleResolver.h"
#include "StyleScope.h"
#include "StyleScopeRuleSets.h"

namespace WebCore {
namespace Style {

template <typename TraverseFunction>
inline void traverseRuleFeaturesInShadowTree(Element& element, TraverseFunction&& function)
{
    if (!element.shadowRoot())
        return;

    auto& shadowRuleSets = element.shadowRoot()->styleScope().resolver().ruleSets();
    bool hasHostPseudoClassRule = shadowRuleSets.hasMatchingUserOrAuthorStyle([&] (auto& style) {
        return !style.hostPseudoClassRules().isEmpty() || style.hasHostPseudoClassRulesMatchingInShadowTree();
    });
    if (!hasHostPseudoClassRule)
        return;

    function(shadowRuleSets.features(), false);
}

template <typename TraverseFunction>
inline void traverseRuleFeaturesForSlotted(Element& element, TraverseFunction&& function)
{
    auto assignedShadowRoots = assignedShadowRootsIfSlotted(element);
    for (auto& assignedShadowRoot : assignedShadowRoots) {
        auto& ruleSets = assignedShadowRoot->styleScope().resolver().ruleSets();
        if (!ruleSets.hasMatchingUserOrAuthorStyle([] (auto& style) { return !style.slottedPseudoElementRules().isEmpty(); }))
            continue;

        function(ruleSets.features(), false);
    }
}

template <typename TraverseFunction>
inline void traverseRuleFeatures(Element& element, TraverseFunction&& function)
{
    auto& ruleSets = element.styleResolver().ruleSets();

    auto mayAffectShadowTree = [&] {
        if (element.shadowRoot() && element.shadowRoot()->isUserAgentShadowRoot()) {
            if (ruleSets.hasMatchingUserOrAuthorStyle([] (auto& style) { return style.hasUserAgentPartRules(); }))
                return true;
#if ENABLE(VIDEO)
            if (element.isMediaElement() && ruleSets.hasMatchingUserOrAuthorStyle([] (auto& style) { return !style.cuePseudoRules().isEmpty(); }))
                return true;
#endif
        }
        return false;
    };

    function(ruleSets.features(), mayAffectShadowTree());

    traverseRuleFeaturesInShadowTree(element, function);
    traverseRuleFeaturesForSlotted(element, function);

    // Ensure that the containing tree resolver also exists so it doesn't get created in the middle of invalidation.
    if (element.isInShadowTree() && element.containingShadowRoot()) {
        auto& host = *element.containingShadowRoot()->host();
        if (host.isConnected())
            Style::Scope::forNode(host).resolver();
    }
}

}
}

