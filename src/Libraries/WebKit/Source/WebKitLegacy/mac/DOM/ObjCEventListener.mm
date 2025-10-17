/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#import "ObjCEventListener.h"

#import "DOMEventInternal.h"
#import "DOMEventListener.h"
#import <WebCore/Event.h>
#import <WebCore/EventListener.h>
#import <WebCore/JSExecState.h>
#import <wtf/HashMap.h>

namespace WebCore {

typedef HashMap<id, ObjCEventListener*> ListenerMap;
static ListenerMap* listenerMap;

ObjCEventListener* ObjCEventListener::find(ObjCListener listener)
{
    ListenerMap* map = listenerMap;
    if (!map)
        return 0;
    return map->get(listener);
}

RefPtr<ObjCEventListener> ObjCEventListener::wrap(ObjCListener listener)
{
    if (!listener)
        return nullptr;
    if (RefPtr<ObjCEventListener> wrapper = find(listener))
        return wrapper;
    return adoptRef(new ObjCEventListener(listener));
}

ObjCEventListener::ObjCEventListener(ObjCListener listener)
    : EventListener(ObjCEventListenerType)
    , m_listener(listener)
{
    ListenerMap* map = listenerMap;
    if (!map) {
        map = new ListenerMap;
        listenerMap = map;
    }
    map->set(listener, this);
}

ObjCEventListener::~ObjCEventListener()
{
    listenerMap->remove(m_listener.get());
    // Avoid executing arbitrary code during GC; e.g. inside Node::~Node. Use CF* to be ARC safe.
    CFRetain((__bridge CFTypeRef)m_listener.get());
    CFAutorelease((__bridge CFTypeRef)m_listener.get());
}

void ObjCEventListener::handleEvent(ScriptExecutionContext&, Event& event)
{
    ObjCListener listener = m_listener.get();
    [listener handleEvent:kit(&event)];
}

bool ObjCEventListener::operator==(const EventListener& listener) const
{
    if (const ObjCEventListener* objCEventListener = ObjCEventListener::cast(&listener))
        return m_listener == objCEventListener->m_listener;
    return false;
}

} // namespace WebCore
