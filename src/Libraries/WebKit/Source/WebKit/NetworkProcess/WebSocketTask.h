/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

#if PLATFORM(COCOA)
#include "WebSocketTaskCocoa.h"
#elif USE(SOUP)
#include "WebSocketTaskSoup.h"
#elif USE(CURL)
#include "WebSocketTaskCurl.h"
#else
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
class WebSocketTask;
}

namespace WebKit {

struct SessionSet;

class WebSocketTask : public CanMakeWeakPtr<WebSocketTask>, public CanMakeCheckedPtr<WebSocketTask> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebSocketTask);
public:
    typedef uint64_t TaskIdentifier;

    void sendString(std::span<const uint8_t>, CompletionHandler<void()>&&) { }
    void sendData(std::span<const uint8_t>, CompletionHandler<void()>&&) { }
    void close(int32_t code, const String& reason) { }

    void cancel() { }
    void resume() { }
    
    SessionSet* sessionSet() { return nullptr; }
};

} // namespace WebKit

#endif
