/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

#include <glib.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RunLoop.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _GSocket GSocket;

namespace WTF {

class GSocketMonitor {
    WTF_MAKE_NONCOPYABLE(GSocketMonitor);
public:
    GSocketMonitor() = default;
    WTF_EXPORT_PRIVATE ~GSocketMonitor();

    WTF_EXPORT_PRIVATE void start(GSocket*, GIOCondition, RunLoop&, Function<gboolean(GIOCondition)>&&);
    WTF_EXPORT_PRIVATE void stop();
    bool isActive() const { return !!m_source; }

private:
    static gboolean socketSourceCallback(GSocket*, GIOCondition, GSocketMonitor*);

    GRefPtr<GSource> m_source;
    GRefPtr<GCancellable> m_cancellable;
    Function<gboolean(GIOCondition)> m_callback;
    bool m_isExecutingCallback { false };
    bool m_shouldDestroyCallback { false };
};

} // namespace WTF

using WTF::GSocketMonitor;
