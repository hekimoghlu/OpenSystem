/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
#include "DatagramSource.h"

#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

DatagramSource::DatagramSource() = default;

DatagramSource::~DatagramSource() = default;

void DatagramSource::receiveDatagram(std::span<const uint8_t> datagram, bool withFin, std::optional<Exception>&& exception)
{
    if (m_isCancelled || m_isClosed)
        return;
    if (exception) {
        controller().error(*exception);
        clean();
        return;
    }
    auto arrayBuffer = ArrayBuffer::tryCreateUninitialized(datagram.size(), 1);
    if (arrayBuffer)
        memcpySpan(arrayBuffer->mutableSpan(), datagram);
    if (!controller().enqueue(WTFMove(arrayBuffer)))
        doCancel();
    if (withFin) {
        m_isClosed = true;
        controller().close();
        clean();
    }
}

void DatagramSource::doCancel()
{
    m_isCancelled = true;
}

}
