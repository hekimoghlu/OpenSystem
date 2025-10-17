/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#include "BaseButtonInputType.h"
#include "IntPoint.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ImageInputType final : public BaseButtonInputType {
    WTF_MAKE_TZONE_ALLOCATED(ImageInputType);
public:
    static Ref<ImageInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new ImageInputType(element));
    }

private:
    explicit ImageInputType(HTMLInputElement&);

    const AtomString& formControlType() const final;
    bool isFormDataAppendable() const final;
    bool appendFormData(DOMFormData&) const final;
    RenderPtr<RenderElement> createInputRenderer(RenderStyle&&) final;
    void handleDOMActivateEvent(Event&) final;
    void attributeChanged(const QualifiedName&) final;
    void attach() final;
    bool shouldRespectAlignAttribute() final;
    bool canBeSuccessfulSubmitButton() final;
    bool shouldRespectHeightAndWidthAttributes() final;
    unsigned height() const final;
    unsigned width() const final;
    String resultForDialogSubmit() const final;
    bool dirAutoUsesValue() const final;

    IntPoint m_clickLocation; // Valid only during HTMLFormElement::submitIfPossible().
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(ImageInputType, Type::Image)
