/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#include "DocumentSyncData.h"

#include "ProcessSyncData.h"
#include <wtf/EnumTraits.h>

namespace WebCore {

void DocumentSyncData::update(const ProcessSyncData& data)
{
    switch (data.type) {
    case ProcessSyncDataType::IsAutofocusProcessed:
        isAutofocusProcessed = std::get<enumToUnderlyingType(ProcessSyncDataType::IsAutofocusProcessed)>(data.value);
        break;
#if ENABLE(DOM_AUDIO_SESSION)
    case ProcessSyncDataType::AudioSessionType:
        audioSessionType = std::get<enumToUnderlyingType(ProcessSyncDataType::AudioSessionType)>(data.value);
        break;
#endif
    case ProcessSyncDataType::UserDidInteractWithPage:
        userDidInteractWithPage = std::get<enumToUnderlyingType(ProcessSyncDataType::UserDidInteractWithPage)>(data.value);
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

DocumentSyncData::DocumentSyncData(
      bool isAutofocusProcessed
#if ENABLE(DOM_AUDIO_SESSION)
    , WebCore::DOMAudioSessionType audioSessionType
#endif
    , bool userDidInteractWithPage
)
    : isAutofocusProcessed(isAutofocusProcessed)
#if ENABLE(DOM_AUDIO_SESSION)
    , audioSessionType(audioSessionType)
#endif
    , userDidInteractWithPage(userDidInteractWithPage)
{
}

} // namespace WebCore
