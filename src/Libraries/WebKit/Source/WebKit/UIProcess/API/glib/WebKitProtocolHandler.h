/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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

#include "WebKitURISchemeRequest.h"
#include "WebKitWebContext.h"
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebKitProtocolHandler {
    WTF_MAKE_TZONE_ALLOCATED(WebKitProtocolHandler);
public:
    explicit WebKitProtocolHandler(WebKitWebContext*);
    ~WebKitProtocolHandler() = default;

private:
    void handleRequest(WebKitURISchemeRequest*);
    void handleGPU(WebKitURISchemeRequest*);
};

} // namespace WebKit
