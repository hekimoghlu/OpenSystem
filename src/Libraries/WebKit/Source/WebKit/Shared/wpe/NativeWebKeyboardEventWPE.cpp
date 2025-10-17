/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#include "config.h"
#include "NativeWebKeyboardEvent.h"

#if ENABLE(WPE_PLATFORM)

#include "WebEventFactory.h"

namespace WebKit {

NativeWebKeyboardEvent::NativeWebKeyboardEvent(WPEEvent* event, const String& text, bool isAutorepeat)
    : WebKeyboardEvent(WebEventFactory::createWebKeyboardEvent(event, text, isAutorepeat))
{
}

NativeWebKeyboardEvent::NativeWebKeyboardEvent(const String& text, std::optional<Vector<WebCore::CompositionUnderline>>&& preeditUnderlines, std::optional<EditingRange>&& preeditSelectionRange)
    : WebKeyboardEvent(WebEvent(WebEventType::KeyDown, { }, WallTime::now()), text, "Unidentified"_s, "Unidentified"_s, "U+0000"_s, 0, 0, true, WTFMove(preeditUnderlines), WTFMove(preeditSelectionRange), false, false)
{
}

} // namespace WebKit

#endif // ENABLE(WPE_PLATFORM)
