/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#include "WebKitWebViewBackend.h"

#include "WebKitWebViewBackendPrivate.h"

/**
 * WebKitWebViewBackend:
 * @See_also: #WebKitWebView.
 *
 * A web view backend.
 *
 * A WebKitWebViewBackend is a boxed type wrapping a WPE backend used to create a
 * #WebKitWebView. A WebKitWebViewBackend is created with webkit_web_view_backend_new()
 * and it should be passed to a WebKitWebView constructor that will take the ownership.
 *
 * Since: 2.20
 */

struct _WebKitWebViewBackend {
    _WebKitWebViewBackend(struct wpe_view_backend* backend, GDestroyNotify notifyCallback, gpointer notifyCallbackData)
        : backend(backend)
        , notifyCallback(notifyCallback)
        , notifyCallbackData(notifyCallbackData)
    {
        ASSERT(backend);
        ASSERT(notifyCallback);
        ASSERT(notifyCallbackData);
    }

    ~_WebKitWebViewBackend()
    {
        notifyCallback(notifyCallbackData);
    }

    struct wpe_view_backend* backend;
    GDestroyNotify notifyCallback;
    gpointer notifyCallbackData;
    int referenceCount { 1 };
};

static WebKitWebViewBackend* webkitWebViewBackendRef(WebKitWebViewBackend* viewBackend)
{
    ASSERT(viewBackend);
    g_atomic_int_inc(&viewBackend->referenceCount);
    return viewBackend;
}

G_DEFINE_BOXED_TYPE(WebKitWebViewBackend, webkit_web_view_backend, webkitWebViewBackendRef, webkitWebViewBackendUnref)

void webkitWebViewBackendUnref(WebKitWebViewBackend* viewBackend)
{
    ASSERT(viewBackend);
    if (g_atomic_int_dec_and_test(&viewBackend->referenceCount)) {
        viewBackend->~WebKitWebViewBackend();
        fastFree(viewBackend);
    }
}

/**
 * webkit_web_view_backend_new:
 * @backend: (transfer full): a #wpe_view_backend
 * @notify: (nullable): a #GDestroyNotify, or %NULL
 * @user_data: user data to pass to @notify
 *
 * Create a new #WebKitWebViewBackend for the given WPE @backend. You can pass a #GDestroyNotify
 * that will be called when the object is destroyed passing @user_data as the argument. If @notify
 * is %NULL, wpe_view_backend_destroy() will be used with @backend as argument.
 * The returned #WebKitWebViewBackend should never be freed by the user; it must be passed to a
 * #WebKitWebView constructor that will take the ownership.
 *
 * Returns: a newly created #WebKitWebViewBackend
 *
 * Since: 2.20
 */
WebKitWebViewBackend* webkit_web_view_backend_new(struct wpe_view_backend* backend, GDestroyNotify notify, gpointer userData)
{
    g_return_val_if_fail(backend, nullptr);

    auto* viewBackend = static_cast<WebKitWebViewBackend*>(fastMalloc(sizeof(WebKitWebViewBackend)));
    new (viewBackend) WebKitWebViewBackend(backend, notify ? notify : reinterpret_cast<GDestroyNotify>(wpe_view_backend_destroy), notify ? userData : backend);
    return viewBackend;
}

/**
 * webkit_web_view_backend_get_wpe_backend:
 * @view_backend: a #WebKitWebViewBackend
 *
 * Get the WPE backend of @view_backend
 *
 * Returns: (transfer none): the #wpe_view_backend
 *
 * Since: 2.20
 */
struct wpe_view_backend* webkit_web_view_backend_get_wpe_backend(WebKitWebViewBackend* viewBackend)
{
    g_return_val_if_fail(viewBackend, nullptr);
    return viewBackend->backend;
}

namespace WTF {

template <> WebKitWebViewBackend* refGPtr(WebKitWebViewBackend* ptr)
{
    if (ptr)
        webkitWebViewBackendRef(ptr);
    return ptr;
}

template <> void derefGPtr(WebKitWebViewBackend* ptr)
{
    if (ptr)
        webkitWebViewBackendUnref(ptr);
}

}
