/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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

#include "HTMLAnchorElement.h"
#include "LayoutRect.h"
#include <memory>

namespace WebCore {

class HTMLImageElement;
class HitTestResult;
class Path;

class HTMLAreaElement final : public HTMLAnchorElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLAreaElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLAreaElement);
public:
    static Ref<HTMLAreaElement> create(const QualifiedName&, Document&);
    ~HTMLAreaElement();

    bool isDefault() const { return m_shape == Shape::Default; }

    bool mapMouseEvent(LayoutPoint location, const LayoutSize&, HitTestResult&);

    // FIXME: Use RenderElement& instead of RenderObject*.
    WEBCORE_EXPORT LayoutRect computeRect(RenderObject*) const;
    Path computePath(RenderObject*) const;
    Path computePathForFocusRing(const LayoutSize& elementSize) const;

    // The parent map's image.
    WEBCORE_EXPORT RefPtr<HTMLImageElement> imageElement() const;
    
private:
    HTMLAreaElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool supportsFocus() const final;
    AtomString target() const final;
    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool isMouseFocusable() const final;
    bool isFocusable() const final;
    RefPtr<Element> focusAppearanceUpdateTarget() final;
    void setFocus(bool, FocusVisibility = FocusVisibility::Invisible) final;

    enum class Shape : uint8_t { Default, Poly, Rect, Circle };
    Path getRegion(const LayoutSize&) const;
    void invalidateCachedRegion();

    std::unique_ptr<Path> m_region;
    Vector<double> m_coords;
    LayoutSize m_lastSize { -1, -1 };
    Shape m_shape { Shape::Rect };
};

} //namespace
