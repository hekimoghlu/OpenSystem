/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#ifndef ServicesController_h
#define ServicesController_h

#if ENABLE(SERVICE_CONTROLS)

#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>

namespace WebKit {

class ServicesController {
    WTF_MAKE_NONCOPYABLE(ServicesController);
    friend NeverDestroyed<ServicesController>;
public:
    static ServicesController& singleton();

    bool hasImageServices() const { return m_hasImageServices; }
    bool hasSelectionServices() const { return m_hasSelectionServices; }
    bool hasRichContentServices() const { return m_hasRichContentServices; }

    void refreshExistingServices(bool refreshImmediately = true);

private:
    ServicesController();

    dispatch_queue_t m_refreshQueue;
    std::atomic_bool m_hasPendingRefresh;

    std::atomic<bool> m_hasImageServices;
    std::atomic<bool> m_hasSelectionServices;
    std::atomic<bool> m_hasRichContentServices;

    bool m_lastSentHasImageServices;
    bool m_lastSentHasSelectionServices;
    bool m_lastSentHasRichContentServices;

    RetainPtr<id> m_extensionWatcher;
    RetainPtr<id> m_uiExtensionWatcher;
};

} // namespace WebKit

#endif // ENABLE(SERVICE_CONTROLS)
#endif // ServicesController_h
