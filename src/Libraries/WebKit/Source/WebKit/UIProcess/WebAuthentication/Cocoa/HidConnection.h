/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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

#if ENABLE(WEB_AUTHN)

#include <pal/spi/cocoa/IOKitSPI.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class HidConnection : public RefCountedAndCanMakeWeakPtr<HidConnection> {
    WTF_MAKE_NONCOPYABLE(HidConnection);
public:
    enum class DataSent : bool { No, Yes };

    using DataSentCallback = CompletionHandler<void(DataSent)>;
    using DataReceivedCallback = Function<void(Vector<uint8_t>&&)>;

    static Ref<HidConnection> create(IOHIDDeviceRef);
    virtual ~HidConnection();

    // Overrided by MockHidConnection.
    virtual void initialize();
    virtual void terminate();
    virtual DataSent sendSync(const Vector<uint8_t>& data);
    // Caller should send data again after callback is invoked to control flow.
    virtual void send(Vector<uint8_t>&& data, DataSentCallback&&);
    void registerDataReceivedCallback(DataReceivedCallback&&);
    void unregisterDataReceivedCallback();
    void invalidateCache() { m_inputReports.clear(); }

    void receiveReport(Vector<uint8_t>&&);

protected:
    explicit HidConnection(IOHIDDeviceRef);
    bool isInitialized() const { return m_isInitialized; }
    void setIsInitialized(bool isInitialized) { m_isInitialized = isInitialized; }

private:
    void consumeReports();

    // Overrided by MockHidConnection.
    virtual void registerDataReceivedCallbackInternal();

    RetainPtr<IOHIDDeviceRef> m_device;
    Vector<uint8_t> m_inputBuffer;
    // Could queue data requested by other applications.
    Deque<Vector<uint8_t>> m_inputReports;
    DataReceivedCallback m_inputCallback;
    bool m_isInitialized { false };
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
