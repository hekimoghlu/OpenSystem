/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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

#if ENABLE(NOTIFICATIONS)

#include "Notification.h"
#include "SharedBuffer.h"
#include "ThreadableLoader.h"
#include <wtf/CompletionHandler.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class BitmapImage;
class NetworkLoadMetrics;
class NotificationResources;
class ResourceError;
class ResourceResponse;

class NotificationResourcesLoader {
    WTF_MAKE_TZONE_ALLOCATED(NotificationResourcesLoader);
public:
    explicit NotificationResourcesLoader(Notification&);

    void start(CompletionHandler<void(RefPtr<NotificationResources>&&)>&&);
    void stop();

private:
    enum class Resource { Image, Icon, Badge, ActionIcon };
    static bool resourceIsSupportedInPlatform(Resource);

    class ResourceLoader final : public ThreadableLoaderClient {
        WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ResourceLoader);
    public:
        ResourceLoader(ScriptExecutionContext&, const URL&, CompletionHandler<void(ResourceLoader*, RefPtr<BitmapImage>&&)>&&);
        ~ResourceLoader();

        void cancel();

        bool finished() const { return m_finished; }

    private:
        // ThreadableLoaderClient API.
        void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) final;
        void didReceiveData(const SharedBuffer&) final;
        void didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&) final;
        void didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&) final;

        bool m_finished { false };
        SharedBufferBuilder m_buffer;
        RefPtr<BitmapImage> m_image;
        RefPtr<ThreadableLoader> m_loader;
        CompletionHandler<void(ResourceLoader*, RefPtr<BitmapImage>&&)> m_completionHandler;
    };

    void didFinishLoadingResource(ResourceLoader*);

    Notification& m_notification;
    bool m_stopped { false };
    CompletionHandler<void(RefPtr<NotificationResources>&&)> m_completionHandler;
    HashSet<std::unique_ptr<ResourceLoader>> m_loaders;
    RefPtr<NotificationResources> m_resources;
};

} // namespace WebCore

#endif // ENABLE(NOTIFICATIONS)
