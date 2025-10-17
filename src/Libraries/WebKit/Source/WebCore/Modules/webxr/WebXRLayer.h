/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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

#if ENABLE(WEBXR)

#include "ContextDestructionObserver.h"
#include "EventTarget.h"
#include "PlatformXR.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ScriptExecutionContext;

class WebXRLayer : public RefCounted<WebXRLayer>, public EventTarget, public ContextDestructionObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRLayer);
public:
    virtual ~WebXRLayer();

    using RefCounted<WebXRLayer>::ref;
    using RefCounted<WebXRLayer>::deref;

    virtual void startFrame(PlatformXR::FrameData&) = 0;
    virtual PlatformXR::Device::Layer endFrame() = 0;

protected:
    explicit WebXRLayer(ScriptExecutionContext*);

    // EventTarget
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }

private:
    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WebXRLayer; }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
