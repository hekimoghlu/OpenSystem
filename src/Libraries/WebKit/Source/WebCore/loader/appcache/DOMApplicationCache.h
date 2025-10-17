/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#include <wtf/WeakPtr.h>

namespace WebCore {

class ApplicationCacheHost;
class LocalFrame;

class DOMApplicationCache final : public RefCounted<DOMApplicationCache>, public EventTarget, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DOMApplicationCache);
public:
    static Ref<DOMApplicationCache> create(LocalDOMWindow& window) { return adoptRef(*new DOMApplicationCache(window)); }
    virtual ~DOMApplicationCache() { ASSERT(!frame()); }

    unsigned short status() const;
    ExceptionOr<void> update();
    ExceptionOr<void> swapCache();
    void abort();

    using RefCounted::ref;
    using RefCounted::deref;

private:
    explicit DOMApplicationCache(LocalDOMWindow&);

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::DOMApplicationCache; }
    ScriptExecutionContext* scriptExecutionContext() const final;

    ApplicationCacheHost* applicationCacheHost() const;
};

} // namespace WebCore
