/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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


namespace WebCore {

class SocketStreamError;
class SocketStreamHandle;

class SocketStreamHandleClient {
public:
    virtual ~SocketStreamHandleClient() = default;

    virtual void didOpenSocketStream(SocketStreamHandle&) = 0;
    virtual void didCloseSocketStream(SocketStreamHandle&) = 0;
    virtual void didReceiveSocketStreamData(SocketStreamHandle&, std::span<const uint8_t> data) = 0;
    virtual void didFailToReceiveSocketStreamData(SocketStreamHandle&) = 0;
    virtual void didUpdateBufferedAmount(SocketStreamHandle&, size_t bufferedAmount) = 0;
    virtual void didFailSocketStream(SocketStreamHandle&, const SocketStreamError&) = 0;
};

} // namespace WebCore
