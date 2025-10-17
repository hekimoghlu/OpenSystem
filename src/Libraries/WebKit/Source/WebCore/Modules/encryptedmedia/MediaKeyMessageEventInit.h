/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "Event.h"
#include "MediaKeyMessageType.h"
#include <JavaScriptCore/ArrayBuffer.h>

namespace WebCore {

struct MediaKeyMessageEventInit : EventInit {
    MediaKeyMessageEventInit() = default;

    MediaKeyMessageEventInit(MediaKeyMessageType messageType, RefPtr<JSC::ArrayBuffer>&& message)
        : EventInit()
        , messageType(messageType)
        , message(WTFMove(message))
    { }

    MediaKeyMessageType messageType;
    RefPtr<JSC::ArrayBuffer> message;
};

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
