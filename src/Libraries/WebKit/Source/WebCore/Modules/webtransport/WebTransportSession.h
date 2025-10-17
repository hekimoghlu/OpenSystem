/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

#include <span>
#include <wtf/AbstractRefCounted.h>
#include <wtf/NativePromise.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class Exception;
class ReadableStreamSource;
class ScriptExecutionContext;
class WebTransportBidirectionalStream;
class WebTransportSendStream;
class WebTransportSessionClient;
class WritableStreamSink;

struct WebTransportBidirectionalStreamConstructionParameters;

using WritableStreamPromise = NativePromise<Ref<WritableStreamSink>, void>;
using BidirectionalStreamPromise = NativePromise<WebTransportBidirectionalStreamConstructionParameters, void>;
using WebTransportSendPromise = NativePromise<std::optional<Exception>, void>;

struct WebTransportStreamIdentifierType;
using WebTransportStreamIdentifier = ObjectIdentifier<WebTransportStreamIdentifierType>;

using WebTransportSessionErrorCode = uint32_t;
using WebTransportStreamErrorCode = uint64_t;

class WEBCORE_EXPORT WebTransportSession : public AbstractRefCounted {
public:
    virtual ~WebTransportSession();

    virtual Ref<WebTransportSendPromise> sendDatagram(std::span<const uint8_t>) = 0;
    virtual Ref<WritableStreamPromise> createOutgoingUnidirectionalStream() = 0;
    virtual Ref<BidirectionalStreamPromise> createBidirectionalStream() = 0;
    virtual void cancelReceiveStream(WebTransportStreamIdentifier, std::optional<WebTransportStreamErrorCode>) = 0;
    virtual void cancelSendStream(WebTransportStreamIdentifier, std::optional<WebTransportStreamErrorCode>) = 0;
    virtual void destroyStream(WebTransportStreamIdentifier, std::optional<WebTransportStreamErrorCode>) = 0;
    virtual void terminate(WebTransportSessionErrorCode, CString&&) = 0;
};

}
