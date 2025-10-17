/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
#include "ZStream.h"

#include <wtf/StdLibExtras.h>

namespace WebCore {

bool ZStream::initializeIfNecessary(Algorithm algorithm, Operation operation)
{
    if (m_isInitialized)
        return true;

    int result = Z_OK;

    switch (operation) {
    case Operation::Compression:
        switch (algorithm) {
        // Values chosen here are based off
        // https://developer.apple.com/documentation/compression/compression_algorithm/compression_zlib?language=objc
        case Algorithm::Deflate:
            result = deflateInit2(&m_stream, 5, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
            break;
        case Algorithm::Zlib:
            result = deflateInit2(&m_stream, 5, Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY);
            break;
        case Algorithm::Gzip:
            result = deflateInit2(&m_stream, 5, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
            break;
        }
        break;
    case Operation::Decompression:
        switch (algorithm) {
        case Algorithm::Deflate:
            result = inflateInit2(&m_stream, -15);
            break;
        case Algorithm::Zlib:
            result = inflateInit2(&m_stream, 15);
            break;
        case Algorithm::Gzip:
            result = inflateInit2(&m_stream, 15 + 16);
            break;
        }
        break;
    }
    if (result != Z_OK)
        return false;
    m_isInitialized = true;
    return true;
}

ZStream::ZStream()
{
    zeroBytes(m_stream);
}

ZStream::~ZStream()
{
    if (m_isInitialized)
        deflateEnd(&m_stream);
}

}
