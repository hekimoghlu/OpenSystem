/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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

#include "ContainerQuery.h"
#include "GenericMediaQueryEvaluator.h"
#include "StyleScopeOrdinal.h"
#include "StyleUpdate.h"
#include <wtf/Ref.h>

namespace WebCore {

class Element;

namespace Style {

struct ContainerQueryEvaluationState {
    Vector<Ref<const Element>> sizeQueryContainers;
    CheckedPtr<Style::Update> styleUpdate;
};

class ContainerQueryEvaluator : public MQ::GenericMediaQueryEvaluator<ContainerQueryEvaluator> {
public:
    enum class SelectionMode : uint8_t { Element, PseudoElement, PartPseudoElement };
    ContainerQueryEvaluator(const Element&, SelectionMode, ScopeOrdinal, ContainerQueryEvaluationState*);

    bool evaluate(const CQ::ContainerQuery&) const;

    static const Element* selectContainer(OptionSet<CQ::Axis>, const String& name, const Element&, SelectionMode = SelectionMode::Element, ScopeOrdinal = ScopeOrdinal::Element, const ContainerQueryEvaluationState* = nullptr);

private:
    std::optional<MQ::FeatureEvaluationContext> featureEvaluationContextForQuery(const CQ::ContainerQuery&) const;

    const Ref<const Element> m_element;
    const SelectionMode m_selectionMode;
    const ScopeOrdinal m_scopeOrdinal;
    ContainerQueryEvaluationState* m_evaluationState { nullptr };
};

}
}
