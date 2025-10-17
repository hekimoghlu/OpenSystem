/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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

#include "BaseCheckableInputType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum class WasSetByJavaScript : bool;

class RadioInputType final : public BaseCheckableInputType {
    WTF_MAKE_TZONE_ALLOCATED(RadioInputType);
public:
    static Ref<RadioInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new RadioInputType(element));
    }

    static void forEachButtonInDetachedGroup(ContainerNode& rootName, const String& groupName, const Function<bool(HTMLInputElement&)>&);

    bool valueMissing(const String&) const final;

private:
    explicit RadioInputType(HTMLInputElement& element)
        : BaseCheckableInputType(Type::Radio, element)
    {
    }

    const AtomString& formControlType() const final;
    String valueMissingText() const final;
    void handleClickEvent(MouseEvent&) final;
    ShouldCallBaseEventHandler handleKeydownEvent(KeyboardEvent&) final;
    void handleKeyupEvent(KeyboardEvent&) final;
    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool shouldSendChangeEventAfterCheckedChanged() final;
    void willDispatchClick(InputElementClickState&) final;
    void didDispatchClick(Event&, const InputElementClickState&) final;
    bool matchesIndeterminatePseudoClass() const final;
    void willUpdateCheckedness(bool /* nowChecked */, WasSetByJavaScript) final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(RadioInputType, Type::Radio)
