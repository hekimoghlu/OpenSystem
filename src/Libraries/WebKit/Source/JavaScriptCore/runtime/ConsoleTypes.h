/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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

#include <wtf/Forward.h>

namespace JSC {

enum class MessageSource : uint8_t {
    XML,
    JS,
    Network,
    ConsoleAPI,
    Storage,
    AppCache,
    Rendering,
    CSS,
    Security,
    ContentBlocker,
    Media,
    MediaSource,
    WebRTC,
    ITPDebug,
    PrivateClickMeasurement,
    PaymentRequest,
    Other,
};

enum class MessageType : uint8_t {
    Log,
    Dir,
    DirXML,
    Table,
    Trace,
    StartGroup,
    StartGroupCollapsed,
    EndGroup,
    Clear,
    Assert,
    Timing,
    Profile,
    ProfileEnd,
    Image,
};

enum class MessageLevel : uint8_t {
    Log,
    Warning,
    Error,
    Debug,
    Info,
};

} // namespace JSC

using JSC::MessageSource;
using JSC::MessageLevel;
