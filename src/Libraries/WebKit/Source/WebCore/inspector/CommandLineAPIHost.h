/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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

#include "InstrumentingAgents.h"
#include <JavaScriptCore/PerGlobalObjectWrapperWorld.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSValue;
}

namespace WebCore {

class Database;
class EventTarget;
class JSDOMGlobalObject;
class Storage;

#if ENABLE(WEB_RTC)
class RTCLogsCallback;
#endif

struct EventListenerInfo;

class CommandLineAPIHost : public RefCounted<CommandLineAPIHost> {
public:
    static Ref<CommandLineAPIHost> create();
    ~CommandLineAPIHost();

    void init(RefPtr<InstrumentingAgents> instrumentingAgents)
    {
        m_instrumentingAgents = instrumentingAgents;
    }

    void disconnect();

    void copyText(const String& text);

    class InspectableObject {
        WTF_MAKE_TZONE_ALLOCATED(InspectableObject);
    public:
        virtual JSC::JSValue get(JSC::JSGlobalObject&);
        virtual ~InspectableObject() = default;
    };
    void addInspectedObject(std::unique_ptr<InspectableObject>);
    JSC::JSValue inspectedObject(JSC::JSGlobalObject&);
    void inspect(JSC::JSGlobalObject&, JSC::JSValue objectToInspect, JSC::JSValue hints);

    struct ListenerEntry {
        JSC::Strong<JSC::JSObject> listener;
        bool useCapture;
        bool passive;
        bool once;
    };

    using EventListenersRecord = Vector<KeyValuePair<String, Vector<ListenerEntry>>>;
    EventListenersRecord getEventListeners(JSC::JSGlobalObject&, EventTarget&);

#if ENABLE(WEB_RTC)
    void gatherRTCLogs(JSC::JSGlobalObject&, RefPtr<RTCLogsCallback>&&);
#endif

    String databaseId(Database&);
    String storageId(Storage&);

    JSC::JSValue wrapper(JSC::JSGlobalObject*, JSDOMGlobalObject*);
    void clearAllWrappers();

private:
    CommandLineAPIHost();

    RefPtr<InstrumentingAgents> m_instrumentingAgents;
    std::unique_ptr<InspectableObject> m_inspectedObject; // $0
    Inspector::PerGlobalObjectWrapperWorld m_wrappers;
};

} // namespace WebCore
