/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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

#include "HTMLDivElement.h"
#include <wtf/Forward.h>

namespace WebCore {

class RenderTextControlInnerBlock;

class TextControlInnerContainer final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextControlInnerContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextControlInnerContainer);
public:
    static Ref<TextControlInnerContainer> create(Document&);

private:
    explicit TextControlInnerContainer(Document&);
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    std::optional<Style::ResolvedStyle> resolveCustomStyle(const Style::ResolutionContext&, const RenderStyle* shadowHostStyle) override;
};

class TextControlInnerElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextControlInnerElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextControlInnerElement);
public:
    static Ref<TextControlInnerElement> create(Document&);

private:
    explicit TextControlInnerElement(Document&);
    std::optional<Style::ResolvedStyle> resolveCustomStyle(const Style::ResolutionContext&, const RenderStyle* shadowHostStyle) override;

    bool isMouseFocusable() const override { return false; }
};

class TextControlInnerTextElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextControlInnerTextElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextControlInnerTextElement);
public:
    static Ref<TextControlInnerTextElement> create(Document&, bool isEditable);

    void defaultEventHandler(Event&) override;

    RenderTextControlInnerBlock* renderer() const;

    inline void updateInnerTextElementEditability(bool isEditable)
    {
        constexpr bool initialization = false;
        updateInnerTextElementEditabilityImpl(isEditable, initialization);
    }

private:
    void updateInnerTextElementEditabilityImpl(bool isEditable, bool initialization);

    explicit TextControlInnerTextElement(Document&);
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    std::optional<Style::ResolvedStyle> resolveCustomStyle(const Style::ResolutionContext&, const RenderStyle* shadowHostStyle) override;
    bool isMouseFocusable() const override { return false; }
    bool isTextControlInnerTextElement() const override { return true; }
};

class TextControlPlaceholderElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextControlPlaceholderElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextControlPlaceholderElement);
public:
    static Ref<TextControlPlaceholderElement> create(Document&);

private:
    explicit TextControlPlaceholderElement(Document&);
    
    std::optional<Style::ResolvedStyle> resolveCustomStyle(const Style::ResolutionContext&, const RenderStyle* shadowHostStyle) override;
};

class SearchFieldResultsButtonElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SearchFieldResultsButtonElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SearchFieldResultsButtonElement);
public:
    static Ref<SearchFieldResultsButtonElement> create(Document&);

    void defaultEventHandler(Event&) override;
#if !PLATFORM(IOS_FAMILY)
    bool willRespondToMouseClickEventsWithEditability(Editability) const override;
#endif

    bool canAdjustStyleForAppearance() const { return m_canAdjustStyleForAppearance; }

private:
    explicit SearchFieldResultsButtonElement(Document&);
    bool isMouseFocusable() const override { return false; }
    std::optional<Style::ResolvedStyle> resolveCustomStyle(const Style::ResolutionContext&, const RenderStyle* shadowHostStyle) override;
    bool isSearchFieldResultsButtonElement() const override { return true; }

    bool m_canAdjustStyleForAppearance { true };
};

class SearchFieldCancelButtonElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SearchFieldCancelButtonElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SearchFieldCancelButtonElement);
public:
    static Ref<SearchFieldCancelButtonElement> create(Document&);

    void defaultEventHandler(Event&) override;
#if !PLATFORM(IOS_FAMILY)
    bool willRespondToMouseClickEventsWithEditability(Editability) const override;
#endif

private:
    explicit SearchFieldCancelButtonElement(Document&);
    bool isMouseFocusable() const override { return false; }
    std::optional<Style::ResolvedStyle> resolveCustomStyle(const Style::ResolutionContext&, const RenderStyle* shadowHostStyle) override;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::TextControlInnerTextElement)
    static bool isType(const WebCore::HTMLElement& element) { return element.isTextControlInnerTextElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* htmlElement = dynamicDowncast<WebCore::HTMLElement>(node);
        return htmlElement && isType(*htmlElement);
    }
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SearchFieldResultsButtonElement)
    static bool isType(const WebCore::HTMLElement& element) { return element.isSearchFieldResultsButtonElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* htmlElement = dynamicDowncast<WebCore::HTMLElement>(node);
        return htmlElement && isType(*htmlElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
