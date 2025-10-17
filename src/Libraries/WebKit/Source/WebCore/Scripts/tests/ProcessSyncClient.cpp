/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#include "ProcessSyncClient.h"

#include "ProcessSyncData.h"
#include <wtf/EnumTraits.h>

namespace WebCore {

#if ENABLE(DOM_AUDIO_SESSION)
void ProcessSyncClient::broadcastAudioSessionTypeToOtherProcesses(const WebCore::DOMAudioSessionType& data)
{
    ProcessSyncDataVariant dataVariant;
    dataVariant.emplace<enumToUnderlyingType(ProcessSyncDataType::AudioSessionType)>(data);
    broadcastProcessSyncDataToOtherProcesses({ ProcessSyncDataType::AudioSessionType, WTFMove(dataVariant) });
}
#endif
void ProcessSyncClient::broadcastMainFrameURLChangeToOtherProcesses(const URL& data)
{
    ProcessSyncDataVariant dataVariant;
    dataVariant.emplace<enumToUnderlyingType(ProcessSyncDataType::MainFrameURLChange)>(data);
    broadcastProcessSyncDataToOtherProcesses({ ProcessSyncDataType::MainFrameURLChange, WTFMove(dataVariant) });
}
void ProcessSyncClient::broadcastIsAutofocusProcessedToOtherProcesses(const bool& data)
{
    ProcessSyncDataVariant dataVariant;
    dataVariant.emplace<enumToUnderlyingType(ProcessSyncDataType::IsAutofocusProcessed)>(data);
    broadcastProcessSyncDataToOtherProcesses({ ProcessSyncDataType::IsAutofocusProcessed, WTFMove(dataVariant) });
}
void ProcessSyncClient::broadcastUserDidInteractWithPageToOtherProcesses(const bool& data)
{
    ProcessSyncDataVariant dataVariant;
    dataVariant.emplace<enumToUnderlyingType(ProcessSyncDataType::UserDidInteractWithPage)>(data);
    broadcastProcessSyncDataToOtherProcesses({ ProcessSyncDataType::UserDidInteractWithPage, WTFMove(dataVariant) });
}

} // namespace WebCore
