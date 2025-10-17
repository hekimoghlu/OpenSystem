/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#include "SpeechRecognitionProvider.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class DummySpeechRecognitionProvider final : public SpeechRecognitionProvider {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DummySpeechRecognitionProvider);
public:
    class DummySpeechRecognitionConnection final : public SpeechRecognitionConnection {
    public:
        static Ref<DummySpeechRecognitionConnection> create()
        {
            return adoptRef(*new DummySpeechRecognitionConnection());
        }
        void registerClient(SpeechRecognitionConnectionClient&) final { }
        void unregisterClient(SpeechRecognitionConnectionClient&) final { }
        void start(SpeechRecognitionConnectionClientIdentifier, const String&, bool, bool, uint64_t, ClientOrigin&&, FrameIdentifier) final { }
        void stop(SpeechRecognitionConnectionClientIdentifier) final { }
        void abort(SpeechRecognitionConnectionClientIdentifier) final { }
        void didReceiveUpdate(SpeechRecognitionUpdate&&) final { }
    };
    DummySpeechRecognitionProvider() = default;
    SpeechRecognitionConnection& speechRecognitionConnection()
    {
        if (!m_connection)
            m_connection = DummySpeechRecognitionConnection::create();
        return *m_connection;
    }
private:
    RefPtr<DummySpeechRecognitionConnection> m_connection;
};

} // namespace WebCore
