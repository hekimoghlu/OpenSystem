/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#include "CompressionStream.h"

#include <wtf/StdLibExtras.h>

namespace WebCore {

CompressionStream::CompressionStream()
{
#if PLATFORM(COCOA)
    zeroBytes(m_stream);
#endif
}

CompressionStream::~CompressionStream()
{
#if PLATFORM(COCOA)
    if (m_isInitialized)
        compression_stream_destroy(&m_stream);
#endif
}

bool CompressionStream::initializeIfNecessary(Algorithm algorithm, Operation operation)
{
    if (m_isInitialized)
        return true;
#if PLATFORM(COCOA)
    switch (algorithm) {
    case Algorithm::Brotli:
        auto result = compression_stream_init(&m_stream, operation == Operation::Compression ? COMPRESSION_STREAM_ENCODE : COMPRESSION_STREAM_DECODE, COMPRESSION_BROTLI);
        if (result != COMPRESSION_STATUS_OK)
            return false;
        break;
    }
#else
    UNUSED_PARAM(algorithm);
    UNUSED_PARAM(operation);
#endif
    m_isInitialized = true;
    return true;
}

}
