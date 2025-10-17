/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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

#include <WebCore/AddEventListenerOptions.h>
#include <WebCore/EventListener.h>
#include <WebCore/EventTarget.h>
#include <wtf/RefPtr.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

typedef struct _GObject GObject;
typedef struct _GClosure GClosure;

namespace WebKit {

class GObjectEventListener : public WebCore::EventListener {
public:

    static bool addEventListener(GObject* target, WebCore::EventTarget* coreTarget, const char* domEventName, GClosure* handler, bool useCapture)
    {
        Ref<GObjectEventListener> listener(adoptRef(*new GObjectEventListener(target, coreTarget, domEventName, handler, useCapture)));
        return coreTarget->addEventListener(AtomString::fromLatin1(domEventName), WTFMove(listener), useCapture);
    }

    static bool removeEventListener(GObject* target, WebCore::EventTarget* coreTarget, const char* domEventName, GClosure* handler, bool useCapture)
    {
        GObjectEventListener key(target, coreTarget, domEventName, handler, useCapture);
        return coreTarget->removeEventListener(AtomString::fromLatin1(domEventName), key, useCapture);
    }

    static void gobjectDestroyedCallback(GObjectEventListener* listener, GObject*)
    {
        listener->gobjectDestroyed();
    }

    static const GObjectEventListener* cast(const WebCore::EventListener* listener)
    {
        return listener->type() == GObjectEventListenerType
            ? static_cast<const GObjectEventListener*>(listener)
            : nullptr;
    }

    bool operator==(const WebCore::EventListener& other) const override;

private:
    GObjectEventListener(GObject*, WebCore::EventTarget*, const char* domEventName, GClosure*, bool capture);
    ~GObjectEventListener();
    void gobjectDestroyed();

    void handleEvent(WebCore::ScriptExecutionContext&, WebCore::Event&) override;

    GObject* m_target;
    // We do not need to keep a reference to the m_coreTarget, because
    // we only use it when the GObject and thus the m_coreTarget object is alive.
    WebCore::EventTarget* m_coreTarget;
    CString m_domEventName;
    GRefPtr<GClosure> m_handler;
    bool m_capture;
};

} // namespace WebKit

