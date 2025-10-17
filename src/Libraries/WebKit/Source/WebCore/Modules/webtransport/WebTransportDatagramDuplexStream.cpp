/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#include "WebTransportDatagramDuplexStream.h"

#include "JSDOMPromise.h"
#include "ReadableStream.h"
#include "WritableStream.h"

namespace WebCore {

Ref<WebTransportDatagramDuplexStream> WebTransportDatagramDuplexStream::create(Ref<ReadableStream>&& readable, Ref<WritableStream>&& writable)
{
    return adoptRef(*new WebTransportDatagramDuplexStream(WTFMove(readable), WTFMove(writable)));
}

WebTransportDatagramDuplexStream::WebTransportDatagramDuplexStream(Ref<ReadableStream>&& readable, Ref<WritableStream>&& writable)
    : m_readable(WTFMove(readable))
    , m_writable(WTFMove(writable))
{
}

WebTransportDatagramDuplexStream::~WebTransportDatagramDuplexStream() = default;

ReadableStream& WebTransportDatagramDuplexStream::readable()
{
    return m_readable.get();
}

WritableStream& WebTransportDatagramDuplexStream::writable()
{
    return m_writable.get();
}

unsigned WebTransportDatagramDuplexStream::maxDatagramSize()
{
    return m_outgoingMaxDatagramSize;
}

double WebTransportDatagramDuplexStream::incomingMaxAge()
{
    return m_incomingDatagramsExpirationDuration;
}

double WebTransportDatagramDuplexStream::outgoingMaxAge()
{
    return m_outgoingDatagramsExpirationDuration;
}

double WebTransportDatagramDuplexStream::incomingHighWaterMark()
{
    return m_incomingDatagramsHighWaterMark;
}

double WebTransportDatagramDuplexStream::outgoingHighWaterMark()
{
    return m_outgoingDatagramsHighWaterMark;
}

ExceptionOr<void> WebTransportDatagramDuplexStream::setIncomingMaxAge(double maxAge)
{
    // https://www.w3.org/TR/webtransport/#dom-webtransportdatagramduplexstream-incomingmaxage
    if (std::isnan(maxAge) || maxAge < 0)
        return Exception { ExceptionCode::RangeError };
    if (!maxAge)
        maxAge = std::numeric_limits<double>::infinity();
    m_incomingDatagramsExpirationDuration = maxAge;
    return { };
}

ExceptionOr<void> WebTransportDatagramDuplexStream::setOutgoingMaxAge(double maxAge)
{
    // https://www.w3.org/TR/webtransport/#dom-webtransportdatagramduplexstream-outgoingmaxage
    if (std::isnan(maxAge) || maxAge < 0)
        return Exception { ExceptionCode::RangeError };
    if (!maxAge)
        maxAge = std::numeric_limits<double>::infinity();
    m_outgoingDatagramsExpirationDuration = maxAge;
    return { };
}

ExceptionOr<void> WebTransportDatagramDuplexStream::setIncomingHighWaterMark(double mark)
{
    // https://www.w3.org/TR/webtransport/#dom-webtransportdatagramduplexstream-incominghighwatermark
    if (std::isnan(mark) || mark < 0)
        return Exception { ExceptionCode::RangeError };
    if (mark < 1)
        mark = 1;
    m_incomingDatagramsHighWaterMark = mark;
    return { };
}

ExceptionOr<void> WebTransportDatagramDuplexStream::setOutgoingHighWaterMark(double mark)
{
    // https://www.w3.org/TR/webtransport/#dom-webtransportdatagramduplexstream-outgoinghighwatermark
    if (std::isnan(mark) || mark < 0)
        return Exception { ExceptionCode::RangeError };
    if (mark < 1)
        mark = 1;
    m_outgoingDatagramsHighWaterMark = mark;
    return { };
}

}
