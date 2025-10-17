/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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

#include <WebCore/EventListener.h>
#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>

@protocol DOMEventListener;

namespace WebCore {

    class ObjCEventListener : public EventListener {
    public:
        typedef id<DOMEventListener> ObjCListener;
        static RefPtr<ObjCEventListener> wrap(ObjCListener);

        static const ObjCEventListener* cast(const EventListener* listener)
        {
            return listener->type() == ObjCEventListenerType
                ? static_cast<const ObjCEventListener*>(listener)
                : nullptr;
        }

    private:
        static ObjCEventListener* find(ObjCListener);

        ObjCEventListener(ObjCListener);
        virtual ~ObjCEventListener();
        bool operator==(const EventListener&) const override;
        void handleEvent(ScriptExecutionContext&, Event&) override;

        RetainPtr<ObjCListener> m_listener;
    };

} // namespace WebCore
