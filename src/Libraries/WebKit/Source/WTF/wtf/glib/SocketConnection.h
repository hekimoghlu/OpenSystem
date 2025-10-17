/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 14, 2024.
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

#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GSocketMonitor.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>

typedef struct _GSocketConnection GSocketConnection;

namespace WTF {

class SocketConnection : public RefCounted<SocketConnection> {
public:
    typedef void (* MessageCallback)(SocketConnection&, GVariant*, gpointer);
    using MessageHandlers = UncheckedKeyHashMap<CString, std::pair<CString, MessageCallback>>;
    static Ref<SocketConnection> create(GRefPtr<GSocketConnection>&& connection, const MessageHandlers& messageHandlers, gpointer userData)
    {
        return adoptRef(*new SocketConnection(WTFMove(connection), messageHandlers, userData));
    }
    WTF_EXPORT_PRIVATE ~SocketConnection();

    WTF_EXPORT_PRIVATE void sendMessage(const char*, GVariant*);

    bool isClosed() const { return !m_connection; }
    WTF_EXPORT_PRIVATE void close();

private:
    WTF_EXPORT_PRIVATE SocketConnection(GRefPtr<GSocketConnection>&&, const MessageHandlers&, gpointer);

    bool read();
    bool readMessage();
    void write();
    void waitForSocketWritability();
    void didClose();

    GRefPtr<GSocketConnection> m_connection;
    const MessageHandlers& m_messageHandlers;
    gpointer m_userData;
    Vector<gchar> m_readBuffer;
    GSocketMonitor m_readMonitor;
    Vector<gchar> m_writeBuffer;
    GSocketMonitor m_writeMonitor;
};

} // namespace WTF

using WTF::SocketConnection;
