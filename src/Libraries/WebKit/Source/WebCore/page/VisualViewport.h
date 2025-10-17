/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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

#include "EventTarget.h"
#include "LocalDOMWindowProperty.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class VisualViewport final : public RefCounted<VisualViewport>, public EventTarget, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(VisualViewport);
public:
    static Ref<VisualViewport> create(LocalDOMWindow& window) { return adoptRef(*new VisualViewport(window)); }

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final;
    ScriptExecutionContext* scriptExecutionContext() const final;
    bool addEventListener(const AtomString& eventType, Ref<EventListener>&&, const AddEventListenerOptions&) final;

    double offsetLeft() const;
    double offsetTop() const;
    double pageLeft() const;
    double pageTop() const;
    double width() const;
    double height() const;
    double scale() const;

    void update();

    using RefCounted::ref;
    using RefCounted::deref;

private:
    explicit VisualViewport(LocalDOMWindow&);

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    void updateFrameLayout() const;

    double m_offsetLeft { 0 };
    double m_offsetTop { 0 };
    double m_pageLeft { 0 };
    double m_pageTop { 0 };
    double m_width { 0 };
    double m_height { 0 };
    double m_scale { 1 };
};

} // namespace WebCore
