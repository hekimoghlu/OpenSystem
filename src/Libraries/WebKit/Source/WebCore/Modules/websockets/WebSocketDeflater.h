/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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

#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

struct z_stream_s;
typedef z_stream_s z_stream;

namespace WebCore {

class WebSocketDeflater {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(WebSocketDeflater, WEBCORE_EXPORT);
public:
    enum ContextTakeOverMode {
        DoNotTakeOverContext,
        TakeOverContext
    };

    explicit WebSocketDeflater(int windowBits, ContextTakeOverMode = TakeOverContext);
    WEBCORE_EXPORT ~WebSocketDeflater();

    bool initialize();
    bool addBytes(std::span<const uint8_t>);
    bool finish();
    size_t size() const { return m_buffer.size(); }
    std::span<const uint8_t> span() const { return m_buffer.span(); }
    void reset();

private:
    int m_windowBits;
    ContextTakeOverMode m_contextTakeOverMode;
    Vector<uint8_t> m_buffer;
    std::unique_ptr<z_stream> m_stream;
};

class WebSocketInflater {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(WebSocketInflater, WEBCORE_EXPORT);
public:
    explicit WebSocketInflater(int windowBits = 15);
    WEBCORE_EXPORT ~WebSocketInflater();

    bool initialize();
    bool addBytes(std::span<const uint8_t>);
    bool finish();
    size_t size() const { return m_buffer.size(); }
    std::span<const uint8_t> span() const { return m_buffer.span(); }
    void reset();

private:
    int m_windowBits;
    Vector<uint8_t> m_buffer;
    std::unique_ptr<z_stream> m_stream;
};

} // namespace WebCore
