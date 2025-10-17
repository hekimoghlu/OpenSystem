/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

#include "WebMouseEvent.h"
#include <WebCore/FloatPoint.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

typedef struct _GdkDevice GdkDevice;
#if USE(GTK4)
typedef struct _GdkEvent GdkEvent;
#else
typedef union _GdkEvent GdkEvent;
#endif

namespace WebKit {

class WebPageProxy;

class PointerLockManager {
    WTF_MAKE_TZONE_ALLOCATED(PointerLockManager);
    WTF_MAKE_NONCOPYABLE(PointerLockManager);
public:
    static std::unique_ptr<PointerLockManager> create(WebPageProxy&, const WebCore::FloatPoint&, const WebCore::FloatPoint&, WebMouseEventButton, unsigned short, OptionSet<WebEventModifier>);
    PointerLockManager(WebPageProxy&, const WebCore::FloatPoint&, const WebCore::FloatPoint&, WebMouseEventButton, unsigned short, OptionSet<WebEventModifier>);
    virtual ~PointerLockManager();

    virtual bool lock();
    virtual bool unlock();
    virtual void didReceiveMotionEvent(const WebCore::FloatPoint&) { };

protected:
    void handleMotion(const WebCore::FloatSize&);

    WebPageProxy& m_webPage;
    WebCore::FloatPoint m_position;
    WebMouseEventButton m_button { WebMouseEventButton::None };
    unsigned short m_buttons { 0 };
    OptionSet<WebEventModifier> m_modifiers;
    WebCore::FloatPoint m_initialPoint;
    GdkDevice* m_device { nullptr };
};

} // namespace WebKit
