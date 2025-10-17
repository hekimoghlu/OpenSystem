/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

#include "BaseTextInputType.h"
#include "Timer.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SearchFieldResultsButtonElement;

class SearchInputType final : public BaseTextInputType {
    WTF_MAKE_TZONE_ALLOCATED(SearchInputType);
public:
    static Ref<SearchInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new SearchInputType(element));
    }

private:
    explicit SearchInputType(HTMLInputElement&);

    void addSearchResult() final;
    void attributeChanged(const QualifiedName&) final;
    RenderPtr<RenderElement> createInputRenderer(RenderStyle&&) final;
    const AtomString& formControlType() const final;
    bool needsContainer() const final;
    void createShadowSubtree() final;
    void removeShadowSubtree() final;
    HTMLElement* resultsButtonElement() const final;
    HTMLElement* cancelButtonElement() const final;
    ShouldCallBaseEventHandler handleKeydownEvent(KeyboardEvent&) final;
    void didSetValueByUserEdit() final;
    bool sizeShouldIncludeDecoration(int defaultSize, int& preferredSize) const final;
    float decorationWidth() const final;
    void setValue(const String&, bool valueChanged, TextFieldEventBehavior, TextControlSetValueSelection) final;

    RefPtr<SearchFieldResultsButtonElement> m_resultsButton;
    RefPtr<HTMLElement> m_cancelButton;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(SearchInputType, Type::Search)
