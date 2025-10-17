/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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

#include "JSExportMacros.h"
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

class FrontendChannel;

class FrontendRouter : public RefCounted<FrontendRouter> {
public:
    JS_EXPORT_PRIVATE static Ref<FrontendRouter> create();

    bool hasFrontends() const { return !m_connections.isEmpty(); }
    JS_EXPORT_PRIVATE bool hasLocalFrontend() const;
    JS_EXPORT_PRIVATE bool hasRemoteFrontend() const;

    unsigned frontendCount() const { return m_connections.size(); }

    JS_EXPORT_PRIVATE void connectFrontend(FrontendChannel&);
    JS_EXPORT_PRIVATE void disconnectFrontend(FrontendChannel&);
    JS_EXPORT_PRIVATE void disconnectAllFrontends();

    JS_EXPORT_PRIVATE void sendEvent(const String& message) const;
    void sendResponse(const String& message) const;

private:
    Vector<FrontendChannel*, 2> m_connections;
};

} // namespace Inspector
