/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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
#include "WebCodecsEncodedVideoChunk.h"

#if ENABLE(WEB_CODECS)

#include <wtf/StdLibExtras.h>

namespace WebCore {

WebCodecsEncodedVideoChunk::WebCodecsEncodedVideoChunk(Init&& init)
    : m_storage { WebCodecsEncodedVideoChunkStorage::create(init.type, init.timestamp, init.duration, init.data.span()) }
{
}

ExceptionOr<void> WebCodecsEncodedVideoChunk::copyTo(BufferSource&& source)
{
    if (source.length() < byteLength())
        return Exception { ExceptionCode::TypeError, "buffer is too small"_s };

    memcpySpan(source.mutableSpan(), span());
    return { };
}

} // namespace WebCore

#endif // ENABLE(WEB_CODECS)
