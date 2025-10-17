/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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

#include "DOMAudioSession.h"
#include <wtf/URL.h>
#include <variant>

namespace WebCore {

enum class ProcessSyncDataType : uint8_t {
#if ENABLE(DOM_AUDIO_SESSION)
    AudioSessionType = 0,
#endif
    MainFrameURLChange = 1,
    IsAutofocusProcessed = 2,
    UserDidInteractWithPage = 3,
};

static const ProcessSyncDataType allDocumentSyncDataTypes[] = {
    ProcessSyncDataType::IsAutofocusProcessed
#if ENABLE(DOM_AUDIO_SESSION)
    , ProcessSyncDataType::AudioSessionType
#endif
    , ProcessSyncDataType::UserDidInteractWithPage
};

#if !ENABLE(DOM_AUDIO_SESSION)
using DOMAudioSessionType = bool;
#endif

using ProcessSyncDataVariant = std::variant<
    WebCore::DOMAudioSessionType,
    URL,
    bool,
    bool
>;

struct ProcessSyncData {
    ProcessSyncDataType type;
    ProcessSyncDataVariant value;
};

}; // namespace WebCore
