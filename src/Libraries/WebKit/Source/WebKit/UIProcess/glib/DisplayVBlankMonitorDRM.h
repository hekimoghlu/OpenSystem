/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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

#if USE(LIBDRM)

#include "DisplayVBlankMonitor.h"
#include <wtf/unix/UnixFileDescriptor.h>

namespace WebKit {

class DisplayVBlankMonitorDRM final : public DisplayVBlankMonitor {
public:
    static std::unique_ptr<DisplayVBlankMonitor> create(PlatformDisplayID);
    DisplayVBlankMonitorDRM(unsigned, WTF::UnixFileDescriptor&&, int);
    ~DisplayVBlankMonitorDRM() = default;

private:
    Type type() const override { return Type::Drm; }
    bool waitForVBlank() const override;

    WTF::UnixFileDescriptor m_fd;
    int m_crtcBitmask { 0 };
};

} // namespace WebKit

#endif // USE(LIBDRM)
