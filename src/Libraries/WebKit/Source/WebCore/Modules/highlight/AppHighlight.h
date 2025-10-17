/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

#if ENABLE(APP_HIGHLIGHTS)

#include "SharedBuffer.h"
#include <wtf/persistence/PersistentCoders.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class CreateNewGroupForHighlight : bool { No, Yes };

enum class HighlightRequestOriginatedInApp : bool { No, Yes };

struct AppHighlight {
    Ref<WebCore::FragmentedSharedBuffer> highlight;
    std::optional<String> text;
    CreateNewGroupForHighlight isNewGroup;
    HighlightRequestOriginatedInApp requestOriginatedInApp;
};

} // namespace WebCore

namespace IPC {

template<typename AsyncReplyResult> struct AsyncReplyError;

template<> struct AsyncReplyError<WebCore::AppHighlight> {
    static WebCore::AppHighlight create() { return { WebCore::FragmentedSharedBuffer::create(), std::nullopt, WebCore::CreateNewGroupForHighlight::No, WebCore::HighlightRequestOriginatedInApp::No }; }
};

} // namespace IPC


#endif // ENABLE(APP_HIGHLIGHTS)
