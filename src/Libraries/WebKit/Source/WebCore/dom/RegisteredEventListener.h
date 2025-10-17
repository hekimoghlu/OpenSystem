/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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

#include "EventListener.h"
#include <wtf/Ref.h>

namespace WebCore {

// https://dom.spec.whatwg.org/#concept-event-listener
class RegisteredEventListener : public RefCounted<RegisteredEventListener> {
public:
    struct Options {
        Options(bool capture = false, bool passive = false, bool once = false)
            : capture(capture)
            , passive(passive)
            , once(once)
        { }

        bool capture;
        bool passive;
        bool once;
    };

    static Ref<RegisteredEventListener> create(Ref<EventListener>&& listener, const Options& options)
    {
        return adoptRef(*new RegisteredEventListener(WTFMove(listener), options));
    }

    EventListener& callback() const { return m_callback; }
    bool useCapture() const { return m_useCapture; }
    bool isPassive() const { return m_isPassive; }
    bool isOnce() const { return m_isOnce; }
    bool wasRemoved() const { return m_wasRemoved; }

    void markAsRemoved() { m_wasRemoved = true; }

private:
    RegisteredEventListener(Ref<EventListener>&& listener, const Options& options)
        : m_useCapture(options.capture)
        , m_isPassive(options.passive)
        , m_isOnce(options.once)
        , m_wasRemoved(false)
        , m_callback(WTFMove(listener))
    {
    }

    bool m_useCapture : 1;
    bool m_isPassive : 1;
    bool m_isOnce : 1;
    bool m_wasRemoved : 1;
    Ref<EventListener> m_callback;
};

} // namespace WebCore
