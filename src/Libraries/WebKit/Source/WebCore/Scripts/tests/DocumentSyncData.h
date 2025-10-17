/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#include <wtf/TZoneMallocInlines.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include "DOMAudioSession.h"

namespace WebCore {

struct ProcessSyncData;

class DocumentSyncData : public RefCounted<DocumentSyncData> {
WTF_MAKE_TZONE_ALLOCATED_INLINE(DocumentSyncData);
public:
    template<typename... Args>
    static Ref<DocumentSyncData> create(Args&&... args)
    {
        return adoptRef(*new DocumentSyncData(std::forward<Args>(args)...));
    }
    static Ref<DocumentSyncData> create() { return adoptRef(*new DocumentSyncData); }
    void update(const ProcessSyncData&);

    bool isAutofocusProcessed = { };
#if ENABLE(DOM_AUDIO_SESSION)
    WebCore::DOMAudioSessionType audioSessionType = { };
#endif
    bool userDidInteractWithPage = { };

private:
    DocumentSyncData() = default;
    WEBCORE_EXPORT DocumentSyncData(
        bool
#if ENABLE(DOM_AUDIO_SESSION)
      , WebCore::DOMAudioSessionType
#endif
      , bool
    );
};

} // namespace WebCore
