/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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

#if ENABLE(VIDEO)

#include "InbandTextTrack.h"

namespace WebCore {

class DataCue;

class InbandDataTextTrack final : public InbandTextTrack {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InbandDataTextTrack);
public:
    static Ref<InbandDataTextTrack> create(ScriptExecutionContext&, InbandTextTrackPrivate&);
    virtual ~InbandDataTextTrack();

private:
    InbandDataTextTrack(ScriptExecutionContext&, InbandTextTrackPrivate&);

    void addDataCue(const MediaTime& start, const MediaTime& end, std::span<const uint8_t>) final;

    bool shouldPurgeCuesFromUnbufferedRanges() const final { return true; }

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "DataCue"_s; }
#endif

#if ENABLE(DATACUE_VALUE)
    void addDataCue(const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&&, const String&) final;
    void updateDataCue(const MediaTime& start, const MediaTime& end, SerializedPlatformDataCue&) final;
    void removeDataCue(const MediaTime& start, const MediaTime& end, SerializedPlatformDataCue&) final;
    ExceptionOr<void> removeCue(TextTrackCue&) final;

    RefPtr<DataCue> findIncompleteCue(const SerializedPlatformDataCue&);

    Vector<RefPtr<DataCue>> m_incompleteCueMap;
#endif
};

} // namespace WebCore

#endif
