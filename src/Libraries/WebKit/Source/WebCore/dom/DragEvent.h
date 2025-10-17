/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

#include "MouseEvent.h"
#include "MouseEventInit.h"

namespace WebCore {

class DataTransfer;

enum class SyntheticClickType : uint8_t;

struct DragEventInit : public MouseEventInit {
    RefPtr<DataTransfer> dataTransfer;
};

class DragEvent : public MouseEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DragEvent);
public:
    using Init = DragEventInit;

    static Ref<DragEvent> create(const AtomString& eventType, DragEventInit&&);
    static Ref<DragEvent> createForBindings();
    static Ref<DragEvent> create(const AtomString& type, CanBubble, IsCancelable, IsComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&&, int detail,
        const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier>, MouseButton, unsigned short buttons,
        EventTarget* relatedTarget, double force, SyntheticClickType, DataTransfer* = nullptr, IsSimulated = IsSimulated::No, IsTrusted = IsTrusted::Yes);

    virtual ~DragEvent();

    DataTransfer* dataTransfer() const { return m_dataTransfer.get(); }

private:
    DragEvent(const AtomString& eventType, DragEventInit&&);
    DragEvent(const AtomString& type, CanBubble, IsCancelable, IsComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&&, int detail,
        const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier>, MouseButton, unsigned short buttons,
        EventTarget* relatedTarget, double force, SyntheticClickType, DataTransfer*, IsSimulated, IsTrusted);
    DragEvent();

    RefPtr<DataTransfer> m_dataTransfer;
};

} // namespace WebCore
