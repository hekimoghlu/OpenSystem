/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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

#include "ExceptionOr.h"
#include <wtf/Deque.h>
#include <wtf/RefCounted.h>

namespace JSC {
class JSGlobalObject;
}

namespace WebCore {

class DOMPromise;
class ReadableStream;
class WritableStream;

class WebTransportDatagramDuplexStream : public RefCounted<WebTransportDatagramDuplexStream> {
public:
    static Ref<WebTransportDatagramDuplexStream> create(Ref<ReadableStream>&&, Ref<WritableStream>&&);
    ~WebTransportDatagramDuplexStream();

    ReadableStream& readable();
    WritableStream& writable();
    unsigned maxDatagramSize();
    double incomingMaxAge();
    double outgoingMaxAge();
    double incomingHighWaterMark();
    double outgoingHighWaterMark();
    ExceptionOr<void> setIncomingMaxAge(double);
    ExceptionOr<void> setOutgoingMaxAge(double);
    ExceptionOr<void> setIncomingHighWaterMark(double);
    ExceptionOr<void> setOutgoingHighWaterMark(double);

private:
    WebTransportDatagramDuplexStream(Ref<ReadableStream>&&, Ref<WritableStream>&&);

    Ref<ReadableStream> m_readable;
    Ref<WritableStream> m_writable;
    Deque<std::pair<Vector<uint8_t>, MonotonicTime>> m_incomingDatagramsQueue;
    RefPtr<DOMPromise> m_incomingDatagramsPullPromise;
    double m_incomingDatagramsHighWaterMark { 1 };
    double m_incomingDatagramsExpirationDuration { std::numeric_limits<double>::infinity() };
    Deque<std::tuple<Vector<uint8_t>, MonotonicTime, Ref<DOMPromise>>> m_outgoingDatagramsQueue;
    double m_outgoingDatagramsHighWaterMark { 1 };
    double m_outgoingDatagramsExpirationDuration { std::numeric_limits<double>::infinity() };
    size_t m_outgoingMaxDatagramSize { 1024 };
};

}
