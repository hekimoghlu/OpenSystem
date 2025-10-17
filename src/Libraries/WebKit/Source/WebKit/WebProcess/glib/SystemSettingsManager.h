/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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

#if PLATFORM(GTK) || PLATFORM(WPE)
#include "MessageReceiver.h"
#include "WebProcessSupplement.h"
#include <WebCore/SystemSettings.h>
#include <wtf/CheckedRef.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {
class WebProcess;

class SystemSettingsManager final : public WebProcessSupplement, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(SystemSettingsManager);
    WTF_MAKE_NONCOPYABLE(SystemSettingsManager);
public:
    explicit SystemSettingsManager(WebProcess&);
    ~SystemSettingsManager();

    void ref() const final;
    void deref() const final;

    static ASCIILiteral supplementName();

private:
    // WebProcessSupplement.
    void initialize(const WebProcessCreationParameters&) override;

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void didChange(const WebCore::SystemSettings::State&);

    CheckedRef<WebProcess> m_process;
};

} // namespace WebKit

#endif // PLATFORM(GTK) || PLATFORM(WPE)
