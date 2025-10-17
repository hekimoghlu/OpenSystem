/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class DocumentSyncData;
struct ProcessSyncData;

class ProcessSyncClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ProcessSyncClient);

public:
    ProcessSyncClient() = default;
    virtual ~ProcessSyncClient() = default;

    virtual void broadcastTopDocumentSyncDataToOtherProcesses(DocumentSyncData&) { }

#if ENABLE(DOM_AUDIO_SESSION)
    void broadcastAudioSessionTypeToOtherProcesses(const WebCore::DOMAudioSessionType&);
#endif
    void broadcastMainFrameURLChangeToOtherProcesses(const URL&);
    void broadcastIsAutofocusProcessedToOtherProcesses(const bool&);
    void broadcastUserDidInteractWithPageToOtherProcesses(const bool&);

protected:
    virtual void broadcastProcessSyncDataToOtherProcesses(const ProcessSyncData&) { }
};

} // namespace WebCore
