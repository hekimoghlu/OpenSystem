/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#include "NotificationResourcesLoader.h"

#if ENABLE(NOTIFICATIONS)

#include "BitmapImage.h"
#include "GraphicsContext.h"
#include "NotificationResources.h"
#include "ResourceRequest.h"
#include "ResourceResponse.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

// 2.5. Resources
// https://notifications.spec.whatwg.org/#resources

WTF_MAKE_TZONE_ALLOCATED_IMPL(NotificationResourcesLoader);

NotificationResourcesLoader::NotificationResourcesLoader(Notification& notification)
    : m_notification(notification)
{
}

bool NotificationResourcesLoader::resourceIsSupportedInPlatform(Resource resource)
{
    switch (resource) {
    case Resource::Icon:
#if PLATFORM(GTK) || PLATFORM(WPE)
        return true;
#else
        return false;
#endif
    case Resource::Image:
    case Resource::Badge:
    case Resource::ActionIcon:
        // FIXME: Implement other resources.
        return false;
    }

    ASSERT_NOT_REACHED();
    return false;
}

void NotificationResourcesLoader::start(CompletionHandler<void(RefPtr<NotificationResources>&&)>&& completionHandler)
{
    m_completionHandler = WTFMove(completionHandler);

    // If the notification platform supports icons, fetch notificationâ€™s icon URL, if icon URL is set.
    if (resourceIsSupportedInPlatform(Resource::Icon)) {
        const URL& iconURL = m_notification.icon();
        if (!iconURL.isEmpty()) {
            auto loader = makeUnique<ResourceLoader>(*m_notification.scriptExecutionContext(), iconURL, [this](ResourceLoader* loader, RefPtr<BitmapImage>&& image) {
                if (m_stopped)
                    return;

                if (image) {
                    if (!m_resources)
                        m_resources = NotificationResources::create();
                    m_resources->setIcon(WTFMove(image));
                }

                didFinishLoadingResource(loader);
            });

            if (!loader->finished())
                m_loaders.add(WTFMove(loader));
        }
    }

    // FIXME: Implement other resources.

    if (m_loaders.isEmpty())
        m_completionHandler(WTFMove(m_resources));
}

void NotificationResourcesLoader::stop()
{
    if (m_stopped)
        return;

    m_stopped = true;

    auto completionHandler = std::exchange(m_completionHandler, nullptr);

    while (!m_loaders.isEmpty()) {
        auto loader = m_loaders.takeAny();
        loader->cancel();
    }

    if (completionHandler)
        completionHandler(nullptr);
}

void NotificationResourcesLoader::didFinishLoadingResource(ResourceLoader* loader)
{
    if (m_loaders.contains(loader)) {
        m_loaders.remove(loader);
        if (m_loaders.isEmpty() && m_completionHandler)
            m_completionHandler(WTFMove(m_resources));
    }
}

NotificationResourcesLoader::ResourceLoader::ResourceLoader(ScriptExecutionContext& context, const URL& url, CompletionHandler<void(ResourceLoader*, RefPtr<BitmapImage>&&)>&& completionHandler)
    : m_completionHandler(WTFMove(completionHandler))
{
    ThreadableLoaderOptions options;
    options.mode = FetchOptions::Mode::Cors;
    options.sendLoadCallbacks = SendCallbackPolicy::SendCallbacks;
    options.dataBufferingPolicy = DataBufferingPolicy::DoNotBufferData;
    options.contentSecurityPolicyEnforcement = context.shouldBypassMainWorldContentSecurityPolicy() ? ContentSecurityPolicyEnforcement::DoNotEnforce : ContentSecurityPolicyEnforcement::EnforceConnectSrcDirective;
    m_loader = ThreadableLoader::create(context, *this, ResourceRequest(url), options);
}

NotificationResourcesLoader::ResourceLoader::~ResourceLoader()
{
}

void NotificationResourcesLoader::ResourceLoader::cancel()
{
    auto completionHandler = std::exchange(m_completionHandler, nullptr);
    m_loader->cancel();
    m_loader = nullptr;
    if (completionHandler)
        completionHandler(this, nullptr);
}

void NotificationResourcesLoader::ResourceLoader::didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse& response)
{
    // If the response's internal response's type is "default", then attempt to decode the resource as image.
    if (response.type() == ResourceResponse::Type::Default)
        m_image = BitmapImage::create();
}

void NotificationResourcesLoader::ResourceLoader::didReceiveData(const SharedBuffer& buffer)
{
    if (m_image) {
        m_buffer.append(buffer);
        m_image->setData(m_buffer.get(), false);
    }
}

void NotificationResourcesLoader::ResourceLoader::didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&)
{
    m_finished = true;

    if (m_image)
        m_image->setData(m_buffer.take(), true);

    if (m_completionHandler)
        m_completionHandler(this, WTFMove(m_image));
}

void NotificationResourcesLoader::ResourceLoader::didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&)
{
    m_finished = true;

    if (m_completionHandler)
        m_completionHandler(this, nullptr);
}

} // namespace WebCore

#endif // ENABLE(NOTIFICATIONS)
