/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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

#if ENABLE(WEB_AUTHN) && HAVE(NEAR_FIELD)

#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>

OBJC_CLASS NFReaderSession;
OBJC_CLASS NSArray;
OBJC_CLASS WKNFReaderSessionDelegate;

namespace WebKit {

class NfcService;

class NfcConnection : public RefCountedAndCanMakeWeakPtr<NfcConnection> {
public:
    static Ref<NfcConnection> create(RetainPtr<NFReaderSession>&&, NfcService&);
    ~NfcConnection();

    Vector<uint8_t> transact(Vector<uint8_t>&& data) const;
    void stop() const;

    // For WKNFReaderSessionDelegate
    void didDetectTags(NSArray *);

private:
    NfcConnection(RetainPtr<NFReaderSession>&&, NfcService&);

    void restartPolling();
    void startPolling();

    RetainPtr<NFReaderSession> m_session;
    RetainPtr<WKNFReaderSessionDelegate> m_delegate;
    WeakPtr<NfcService> m_service;
    RunLoop::Timer m_retryTimer;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN) && HAVE(NEAR_FIELD)
