/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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

#include "WebSocketDeflater.h"
#include "WebSocketExtensionProcessor.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WebSocketDeflateFramer;
class WebSocketExtensionProcessor;

struct WebSocketFrame;

class DeflateResultHolder {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(DeflateResultHolder, WEBCORE_EXPORT);
public:
    explicit DeflateResultHolder(WebSocketDeflateFramer&);
    WEBCORE_EXPORT ~DeflateResultHolder();

    bool succeeded() const { return m_succeeded; }
    String failureReason() const { return m_failureReason; }

    void fail(const String& failureReason);

private:
    WebSocketDeflateFramer& m_framer;
    bool m_succeeded { true };
    String m_failureReason;
};

class InflateResultHolder {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(InflateResultHolder, WEBCORE_EXPORT);
public:
    explicit InflateResultHolder(WebSocketDeflateFramer&);
    WEBCORE_EXPORT ~InflateResultHolder();

    bool succeeded() const { return m_succeeded; }
    String failureReason() const { return m_failureReason; }

    void fail(const String& failureReason);

private:
    WebSocketDeflateFramer& m_framer;
    bool m_succeeded { true };
    String m_failureReason;
};

class WebSocketDeflateFramer {
public:
    WEBCORE_EXPORT std::unique_ptr<WebSocketExtensionProcessor> createExtensionProcessor();

    bool enabled() const { return m_enabled; }

    WEBCORE_EXPORT std::unique_ptr<DeflateResultHolder> deflate(WebSocketFrame&);
    void resetDeflateContext();
    WEBCORE_EXPORT std::unique_ptr<InflateResultHolder> inflate(WebSocketFrame&);
    void resetInflateContext();

    WEBCORE_EXPORT void didFail();

    void enableDeflate(int windowBits, WebSocketDeflater::ContextTakeOverMode);

private:
    bool m_enabled { false };
    std::unique_ptr<WebSocketDeflater> m_deflater;
    std::unique_ptr<WebSocketInflater> m_inflater;
};

} // namespace WebCore
