/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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

#import "config.h"
#import "BundleResourceLoader.h"

#import "HTTPHeaderMap.h"
#import "MIMETypeRegistry.h"
#import "ResourceError.h"
#import "ResourceLoader.h"
#import "ResourceResponse.h"
#import "SharedBuffer.h"
#import <wtf/RunLoop.h>
#import <wtf/WorkQueue.h>

namespace WebCore {
namespace BundleResourceLoader {

static WorkQueue& loadQueue()
{
    static NeverDestroyed<Ref<WorkQueue>> queue(WorkQueue::create("org.WebKit.BundleResourceLoader"_s, WorkQueue::QOS::Utility));
    return queue.get();
}

void loadResourceFromBundle(ResourceLoader& loader, const String& subdirectory)
{
    ASSERT(RunLoop::isMain());

    loadQueue().dispatch([protectedLoader = Ref { loader }, url = loader.request().url().isolatedCopy(), subdirectory = subdirectory.isolatedCopy()]() mutable {
        auto *relativePath = [subdirectory stringByAppendingString: url.path().toString()];
        auto *bundle = [NSBundle bundleWithIdentifier:@"com.apple.WebCore"];
        auto *path = [bundle pathForResource:relativePath ofType:nil];
        auto *data = [NSData dataWithContentsOfFile:path];

        if (!data) {
            RunLoop::main().dispatch([protectedLoader = WTFMove(protectedLoader), url = WTFMove(url).isolatedCopy()] {
                protectedLoader->didFail(ResourceError { errorDomainWebKitInternal, 0, url, "URL is invalid"_s });
            });
            return;
        }

        RunLoop::main().dispatch([protectedLoader = WTFMove(protectedLoader), url = WTFMove(url).isolatedCopy(), buffer = SharedBuffer::create(data)]() mutable {
            auto mimeType = MIMETypeRegistry::mimeTypeForPath(url.path());
            ResourceResponse response { url, mimeType, static_cast<long long>(buffer->size()), MIMETypeRegistry::isTextMIMEType(mimeType) ? "UTF-8"_s : String() };
            response.setHTTPStatusCode(200);
            response.setHTTPStatusText("OK"_s);
            response.setSource(ResourceResponse::Source::Network);

            // Allow images to load.
            response.addHTTPHeaderField(HTTPHeaderName::AccessControlAllowOrigin, "*"_s);

            protectedLoader->deliverResponseAndData(response, WTFMove(buffer));
        });
    });
}
}
}
