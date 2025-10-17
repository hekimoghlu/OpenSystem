/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

#include "ActiveDOMObject.h"
#include "ReadableStreamSource.h"

namespace JSC {
class ArrayBuffer;
};

namespace WebCore {

class FetchBodyOwner;

class FetchBodySource final : public RefCountedReadableStreamSource {
public:
    FetchBodySource(FetchBodyOwner&);
    virtual ~FetchBodySource();

    bool enqueue(RefPtr<JSC::ArrayBuffer>&& chunk) { return controller().enqueue(WTFMove(chunk)); }
    void close();
    void error(const Exception&);

    bool isCancelling() const { return m_isCancelling; }

    void resolvePullPromise() { pullFinished(); }
    void detach() { m_bodyOwner = nullptr; }

private:
    void doStart() final;
    void doPull() final;
    void doCancel() final;
    void setActive() final;
    void setInactive() final;

    WeakPtr<FetchBodyOwner> m_bodyOwner;

    bool m_isCancelling { false };
#if ASSERT_ENABLED
    bool m_isClosed { false };
#endif
    RefPtr<ActiveDOMObject::PendingActivity<FetchBodyOwner>> m_pendingActivity;
};

} // namespace WebCore
