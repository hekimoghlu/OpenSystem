/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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

#include <JavaScriptCore/InspectorAgentBase.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class BackendDispatcher;
class FrontendRouter;
}

namespace WebKit {

class WebPageProxy;

// FIXME: move this to Inspector namespace when remaining agents move.
struct WebPageAgentContext {
    Inspector::FrontendRouter& frontendRouter;
    Inspector::BackendDispatcher& backendDispatcher;
    WeakRef<WebPageProxy> inspectedPage;
};

class InspectorAgentBase : public Inspector::InspectorAgentBase {
protected:
    InspectorAgentBase(const String& name, WebPageAgentContext&)
        : Inspector::InspectorAgentBase(name)
    {
    }
};

} // namespace WebKit
