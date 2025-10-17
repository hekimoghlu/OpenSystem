/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

#include "WebPageProxyIdentifier.h"
#include <WebCore/PageIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class FrameProcess;
class VisitedLinkStore;
class WebFrameProxy;
class WebProcessProxy;

class ProvisionalFrameProxy {
    WTF_MAKE_TZONE_ALLOCATED(ProvisionalFrameProxy);
public:
    explicit ProvisionalFrameProxy(WebFrameProxy&, Ref<FrameProcess>&&);

    ~ProvisionalFrameProxy();

    WebProcessProxy& process() const;
    Ref<WebProcessProxy> protectedProcess() const;

    RefPtr<FrameProcess> takeFrameProcess();

private:
    WeakRef<WebFrameProxy> m_frame;
    RefPtr<FrameProcess> m_frameProcess;
    Ref<VisitedLinkStore> m_visitedLinkStore;
};

} // namespace WebKit
