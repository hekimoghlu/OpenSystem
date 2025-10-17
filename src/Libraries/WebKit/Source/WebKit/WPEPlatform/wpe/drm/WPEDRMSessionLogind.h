/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#if ENABLE(JOURNALD_LOG)
#include "WPEDRMSession.h"
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GUniquePtr.h>

namespace WPE {

namespace DRM {

class SessionLogind final : public Session {
public:
    static std::unique_ptr<Session> create();
    SessionLogind(GRefPtr<GDBusProxy>&&, GUniquePtr<char>&&);
    ~SessionLogind();

private:
    const char* seatID() const final { return m_seatID.get(); }
    int openDevice(const char*, int) final;
    int closeDevice(int) final;

    GRefPtr<GDBusProxy> m_sessionProxy;
    GUniquePtr<char> m_seatID;
    bool m_inControl { false };
};

} // namespace DRM

} // namespace WPE

#endif // ENABLE(JOURNALD_LOG)
