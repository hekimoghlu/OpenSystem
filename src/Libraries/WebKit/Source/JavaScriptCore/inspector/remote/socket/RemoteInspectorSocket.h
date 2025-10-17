/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "JSExportMacros.h"
#include <array>
#include <wtf/Vector.h>

#if OS(WINDOWS)
#include <winsock2.h>
#else
#include <poll.h>
#endif

namespace Inspector {

using ConnectionID = uint32_t;

#if OS(WINDOWS)

using PlatformSocketType = SOCKET;
using PollingDescriptor = WSAPOLLFD;
constexpr PlatformSocketType INVALID_SOCKET_VALUE = INVALID_SOCKET;

#else

using PlatformSocketType = int;
using PollingDescriptor = struct pollfd;
constexpr PlatformSocketType INVALID_SOCKET_VALUE = -1;

#endif

namespace Socket {

enum class Domain {
    Local,
    Network,
};

void init();

JS_EXPORT_PRIVATE std::optional<PlatformSocketType> connect(const char* serverAddress, uint16_t serverPort);
JS_EXPORT_PRIVATE std::optional<PlatformSocketType> listen(const char* address, uint16_t port);
JS_EXPORT_PRIVATE std::optional<PlatformSocketType> accept(PlatformSocketType);
JS_EXPORT_PRIVATE std::optional<std::array<PlatformSocketType, 2>> createPair();

JS_EXPORT_PRIVATE bool setup(PlatformSocketType);
JS_EXPORT_PRIVATE bool isValid(PlatformSocketType);
JS_EXPORT_PRIVATE bool isListening(PlatformSocketType);
JS_EXPORT_PRIVATE std::optional<uint16_t> getPort(PlatformSocketType);

JS_EXPORT_PRIVATE std::optional<size_t> read(PlatformSocketType, void* buffer, int bufferSize);
JS_EXPORT_PRIVATE std::optional<size_t> write(PlatformSocketType, const void* data, int size);

JS_EXPORT_PRIVATE void close(PlatformSocketType&);

PollingDescriptor preparePolling(PlatformSocketType);
bool poll(Vector<PollingDescriptor>&, int timeout);
bool isReadable(const PollingDescriptor&);
bool isWritable(const PollingDescriptor&);
void markWaitingWritable(PollingDescriptor&);
void clearWaitingWritable(PollingDescriptor&);

constexpr size_t BufferSize = 65536;

} // namespace Socket

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
