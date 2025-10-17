/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#include "BaseClickableWithKeyInputType.h"
#include "ColorChooser.h"
#include "ColorChooserClient.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ColorInputType final : public BaseClickableWithKeyInputType, private ColorChooserClient {
    WTF_MAKE_TZONE_ALLOCATED(ColorInputType);
public:
    static Ref<ColorInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new ColorInputType(element));
    }

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    bool supportsAlpha() const final;
    Vector<Color> suggestedColors() const final;
    Color valueAsColor() const;
    void selectColor(StringView);

    virtual ~ColorInputType();

    void detach() final;

private:
    explicit ColorInputType(HTMLInputElement& element)
        : BaseClickableWithKeyInputType(Type::Color, element)
    {
        ASSERT(needsShadowSubtree());
    }

    void didChooseColor(const Color&) final;
    void didEndChooser() final;
    IntRect elementRectRelativeToRootView() const final;
    bool isMouseFocusable() const final;
    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool isPresentingAttachedView() const final;
    const AtomString& formControlType() const final;
    bool supportsRequired() const final;
    String fallbackValue() const final;
    String sanitizeValue(const String&) const final;
    void createShadowSubtree() final;
    void setValue(const String&, bool valueChanged, TextFieldEventBehavior, TextControlSetValueSelection) final;
    void attributeChanged(const QualifiedName&) final;
    void handleDOMActivateEvent(Event&) final;
    void showPicker() final;
    bool allowsShowPickerAcrossFrames() final;
    void elementDidBlur() final;
    bool shouldRespectListAttribute() final;
    bool shouldResetOnDocumentActivation() final;

    void endColorChooser();
    void updateColorSwatch();
    HTMLElement* shadowColorSwatch() const;

    RefPtr<ColorChooser> m_chooser;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(ColorInputType, Type::Color)
