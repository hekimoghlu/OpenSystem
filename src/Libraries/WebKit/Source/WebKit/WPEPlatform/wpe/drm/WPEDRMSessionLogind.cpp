/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#include "WPEDRMSessionLogind.h"

#if ENABLE(JOURNALD_LOG)
#include <errno.h>
#include <gio/gunixfdlist.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <systemd/sd-login.h>
#include <unistd.h>
#include <wtf/Seconds.h>

namespace WPE {

namespace DRM {

std::unique_ptr<Session> SessionLogind::create()
{
    GUniqueOutPtr<char> session;
    int retval = sd_pid_get_session(getpid(), &session.outPtr());
    if (retval < 0) {
        if (retval != -ENODATA)
            return nullptr;

        // Not inside a systemd session, look if there is a suitable one.
        if (sd_uid_get_display(getuid(), &session.outPtr()) < 0)
            return nullptr;
    }

    GUniqueOutPtr<char> seat;
    if (sd_session_get_seat(session.get(), &seat.outPtr()) < 0)
        return nullptr;

    GUniquePtr<char> path(g_strdup_printf("/org/freedesktop/login1/session/%s", session.get()));
    GRefPtr<GDBusProxy> sessionProxy = adoptGRef(g_dbus_proxy_new_for_bus_sync(G_BUS_TYPE_SYSTEM, G_DBUS_PROXY_FLAGS_DO_NOT_AUTO_START, nullptr, "org.freedesktop.login1", path.get(), "org.freedesktop.login1.Session", nullptr, nullptr));
    return makeUnique<SessionLogind>(WTFMove(sessionProxy), GUniquePtr<char>(seat.release()));
}

SessionLogind::SessionLogind(GRefPtr<GDBusProxy>&& sessionProxy, GUniquePtr<char>&& seatID)
    : m_sessionProxy(WTFMove(sessionProxy))
    , m_seatID(WTFMove(seatID))
{
    GUniqueOutPtr<GError> error;
    GRefPtr<GVariant> result = adoptGRef(g_dbus_proxy_call_sync(m_sessionProxy.get(), "TakeControl", g_variant_new("(b)", FALSE),
        G_DBUS_CALL_FLAGS_NONE, -1, nullptr, &error.outPtr()));
    if (!result) {
        g_warning("Failed to take control of session: %s", error->message);
        return;
    }

    m_inControl = true;
    g_dbus_proxy_call(m_sessionProxy.get(), "Activate", nullptr, G_DBUS_CALL_FLAGS_NONE, -1, nullptr, nullptr, nullptr);
}

SessionLogind::~SessionLogind()
{
    if (!m_inControl)
        return;

    g_dbus_proxy_call(m_sessionProxy.get(), "ReleaseControl", nullptr, G_DBUS_CALL_FLAGS_NONE, -1, nullptr, nullptr, nullptr);
}

int SessionLogind::openDevice(const char* path, int flags)
{
    if (!m_inControl)
        return Session::openDevice(path, flags);

    struct stat st;
    int retval = stat(path, &st);
    if (retval < 0)
        return -1;

    if (!S_ISCHR(st.st_mode)) {
        errno = ENODEV;
        return -1;
    }

    GRefPtr<GUnixFDList> fdList;
    GUniqueOutPtr<GError> error;
    GRefPtr<GVariant> result = adoptGRef(g_dbus_proxy_call_with_unix_fd_list_sync(m_sessionProxy.get(), "TakeDevice", g_variant_new("(uu)", major(st.st_rdev), minor(st.st_rdev)),
        G_DBUS_CALL_FLAGS_NONE, -1, nullptr, &fdList.outPtr(), nullptr, &error.outPtr()));
    if (!result) {
        g_warning("Session failed to take device %s: %s", path, error->message);
        errno = ENODEV;
        return -1;
    }

    int handler;
    g_variant_get(result.get(), "(hb)", &handler, nullptr);
    return g_unix_fd_list_get(fdList.get(), handler, nullptr);
}

int SessionLogind::closeDevice(int deviceID)
{
    if (!m_inControl)
        return Session::closeDevice(deviceID);

    struct stat st;
    int retval = fstat(deviceID, &st);
    if (retval < 0)
        return -1;

    if (!S_ISCHR(st.st_mode)) {
        errno = ENODEV;
        return -1;
    }

    GUniqueOutPtr<GError> error;
    GRefPtr<GVariant> result = adoptGRef(g_dbus_proxy_call_sync(m_sessionProxy.get(), "ReleaseDevice", g_variant_new("(uu)", major(st.st_rdev), minor(st.st_rdev)),
        G_DBUS_CALL_FLAGS_NONE, -1, nullptr, &error.outPtr()));
    if (!result) {
        g_warning("Session failed to release device %d: %s", deviceID, error->message);
        errno = ENODEV;
        return -1;
    }

    return 0;
}

} // namespace DRM

} // namespace WPE

#endif // ENABLE(JOURNALD_LOG)
