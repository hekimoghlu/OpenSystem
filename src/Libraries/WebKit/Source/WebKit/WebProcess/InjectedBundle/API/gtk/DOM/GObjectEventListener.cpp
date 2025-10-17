/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#include "config.h"
#include "GObjectEventListener.h"

#include "WebKitDOMEvent.h"
#include "WebKitDOMEventPrivate.h"
#include "WebKitDOMEventTarget.h"
#include <WebCore/Event.h>
#include <wtf/HashMap.h>

namespace WebKit {
using namespace WebCore;

GObjectEventListener::GObjectEventListener(GObject* target, EventTarget* coreTarget, const char* domEventName, GClosure* handler, bool capture)
    : EventListener(GObjectEventListenerType)
    , m_target(target)
    , m_coreTarget(coreTarget)
    , m_domEventName(domEventName)
    , m_handler(handler)
    , m_capture(capture)
{
    ASSERT(m_coreTarget);
    if (G_CLOSURE_NEEDS_MARSHAL(m_handler.get()))
        g_closure_set_marshal(m_handler.get(), g_cclosure_marshal_generic);
    g_object_weak_ref(m_target, reinterpret_cast<GWeakNotify>(GObjectEventListener::gobjectDestroyedCallback), this);
}

GObjectEventListener::~GObjectEventListener()
{
    if (!m_coreTarget)
        return;
    g_object_weak_unref(m_target, reinterpret_cast<GWeakNotify>(GObjectEventListener::gobjectDestroyedCallback), this);
}

void GObjectEventListener::gobjectDestroyed()
{
    ASSERT(m_coreTarget);

    // Protect 'this' class in case the 'm_coreTarget' holds the last reference,
    // which may cause, inside removeEventListener(), free of this object
    // and later use-after-free with the m_handler = 0; assignment.
    RefPtr<GObjectEventListener> protectedThis(this);

    m_coreTarget->removeEventListener(AtomString::fromLatin1(m_domEventName.data()), *this, m_capture);
    m_coreTarget = nullptr;
    m_handler = nullptr;
}

void GObjectEventListener::handleEvent(ScriptExecutionContext&, Event& event)
{
    G_GNUC_BEGIN_IGNORE_DEPRECATIONS;
    GValue parameters[2] = { G_VALUE_INIT, G_VALUE_INIT };
    g_value_init(&parameters[0], WEBKIT_DOM_TYPE_EVENT_TARGET);
    g_value_set_object(&parameters[0], m_target);

    GRefPtr<WebKitDOMEvent> domEvent = adoptGRef(WebKit::kit(&event));
    g_value_init(&parameters[1], WEBKIT_DOM_TYPE_EVENT);
    g_value_set_object(&parameters[1], domEvent.get());

    g_closure_invoke(m_handler.get(), 0, 2, parameters, NULL);
    g_value_unset(parameters + 0);
    g_value_unset(parameters + 1);
    G_GNUC_END_IGNORE_DEPRECATIONS;
}

bool GObjectEventListener::operator==(const EventListener& listener) const
{
    if (const GObjectEventListener* gobjectEventListener = GObjectEventListener::cast(&listener))
        return m_target == gobjectEventListener->m_target
            && reinterpret_cast<GCClosure*>(m_handler.get())->callback == reinterpret_cast<GCClosure*>(gobjectEventListener->m_handler.get())->callback;

    return false;
}

} // namespace WebKit
