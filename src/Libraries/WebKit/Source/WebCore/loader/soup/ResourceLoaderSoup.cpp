/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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
#include "ResourceLoader.h"

#include "HTTPParsers.h"
#include "ResourceError.h"
#include "SharedBuffer.h"
#include <gio/gio.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/glib/RunLoopSourcePriority.h>

namespace WebCore {

void ResourceLoader::loadGResource()
{
    RefPtr<ResourceLoader> protectedThis(this);
    GRefPtr<GTask> task = adoptGRef(g_task_new(nullptr, nullptr, [](GObject*, GAsyncResult* result, gpointer userData) {
        RefPtr<ResourceLoader> loader = adoptRef(static_cast<ResourceLoader*>(userData));
        if (loader->reachedTerminalState())
            return;

        auto* task = G_TASK(result);
        URL url({ }, String::fromUTF8(static_cast<const char*>(g_task_get_task_data(task))));

        GUniqueOutPtr<GError> error;
        GRefPtr<GBytes> bytes = adoptGRef(static_cast<GBytes*>(g_task_propagate_pointer(task, &error.outPtr())));
        if (!bytes) {
            loader->didFail(ResourceError(String::fromLatin1(g_quark_to_string(error->domain)), error->code, url, String::fromUTF8(error->message)));
            return;
        }

        if (loader->wasCancelled())
            return;

        gsize dataSize;
        const auto* data = static_cast<const guchar*>(g_bytes_get_data(bytes.get(), &dataSize));
        GUniquePtr<char> fileName(g_path_get_basename(url.path().utf8().data()));
        GUniquePtr<char> contentType(g_content_type_guess(fileName.get(), data, dataSize, nullptr));
        auto contentTypeString = String::fromLatin1(contentType.get());
        ResourceResponse response { url, extractMIMETypeFromMediaType(contentTypeString), static_cast<long long>(dataSize), extractCharsetFromMediaType(contentTypeString).toString() };
        response.setHTTPStatusCode(200);
        response.setHTTPStatusText("OK"_s);
        response.setHTTPHeaderField(HTTPHeaderName::ContentType, contentTypeString);
        response.setSource(ResourceResponse::Source::Network);
        loader->deliverResponseAndData(response, SharedBuffer::create(bytes.get()));
    }, protectedThis.leakRef()));

    g_task_set_priority(task.get(), RunLoopSourcePriority::AsyncIONetwork);
    g_task_set_task_data(task.get(), g_strdup(m_request.url().string().utf8().data()), g_free);
    g_task_run_in_thread(task.get(), [](GTask* task, gpointer, gpointer taskData, GCancellable*) {
        URL url({ }, String::fromUTF8(static_cast<const char*>(taskData)));
        GError* error = nullptr;
        GBytes* bytes = g_resources_lookup_data(url.protocolIs("webkit-pdfjs-viewer"_s) ? makeString("/org/webkit/pdfjs"_s, url.path()).utf8().data() : url.path().utf8().data(),
            G_RESOURCE_LOOKUP_FLAGS_NONE, &error);
        if (!bytes)
            g_task_return_error(task, error);
        else
            g_task_return_pointer(task, bytes, reinterpret_cast<GDestroyNotify>(g_bytes_unref));
    });
}

} // namespace WebCore
