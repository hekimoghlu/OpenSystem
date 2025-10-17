/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#include "WebKitPermissionStateQuery.h"

#include "APISecurityOrigin.h"
#include "WebKitPermissionStateQueryPrivate.h"
#include "WebKitSecurityOriginPrivate.h"
#include <wtf/glib/WTFGType.h>

/**
 * WebKitPermissionStateQuery:
 * @See_also: #WebKitWebView
 *
 * This query represents a user's choice to allow or deny access to "powerful features" of the
 * platform, as specified in the [Permissions W3C
 * Specification](https://w3c.github.io/permissions/).
 *
 * When signalled by the #WebKitWebView through the `query-permission-state` signal, the application
 * has to eventually respond, via `webkit_permission_state_query_finish()`, whether it grants,
 * denies or requests a dedicated permission prompt for the given query.
 *
 * When a #WebKitPermissionStateQuery is not handled by the user, the user-agent is instructed to
 * `prompt` the user for the given permission.
 */

struct _WebKitPermissionStateQuery {
    explicit _WebKitPermissionStateQuery(const WTF::String& permissionName, API::SecurityOrigin& origin, CompletionHandler<void(std::optional<WebCore::PermissionState>)>&& completionHandler)
        : permissionName(permissionName.utf8())
        , securityOrigin(webkitSecurityOriginCreate(origin.securityOrigin().isolatedCopy()))
        , completionHandler(WTFMove(completionHandler))
    {
    }

    ~_WebKitPermissionStateQuery()
    {
        // Fallback to Prompt response unless the completion handler was already called.
        if (completionHandler)
            completionHandler(WebCore::PermissionState::Prompt);

        webkit_security_origin_unref(securityOrigin);
    }

    CString permissionName;
    WebKitSecurityOrigin* securityOrigin;
    CompletionHandler<void(std::optional<WebCore::PermissionState>)> completionHandler;
    int referenceCount { 1 };
};

G_DEFINE_BOXED_TYPE(WebKitPermissionStateQuery, webkit_permission_state_query, webkit_permission_state_query_ref, webkit_permission_state_query_unref)

WebKitPermissionStateQuery* webkitPermissionStateQueryCreate(const WTF::String& permissionName, API::SecurityOrigin& origin, CompletionHandler<void(std::optional<WebCore::PermissionState>)>&& completionHandler)
{
    WebKitPermissionStateQuery* query = static_cast<WebKitPermissionStateQuery*>(fastMalloc(sizeof(WebKitPermissionStateQuery)));
    new (query) WebKitPermissionStateQuery(permissionName, origin, WTFMove(completionHandler));
    return query;
}

/**
 * webkit_permission_state_query_ref:
 * @query: a #WebKitPermissionStateQuery
 *
 * Atomically increments the reference count of @query by one.
 *
 * This function is MT-safe and may be called from any thread.
 *
 * Returns: The passed #WebKitPermissionStateQuery
 *
 * Since: 2.40
 */
WebKitPermissionStateQuery* webkit_permission_state_query_ref(WebKitPermissionStateQuery* query)
{
    g_return_val_if_fail(query, nullptr);

    g_atomic_int_inc(&query->referenceCount);
    return query;
}

/**
 * webkit_permission_state_query_unref:
 * @query: a #WebKitPermissionStateQuery
 *
 * Atomically decrements the reference count of @query by one.
 *
 * If the reference count drops to 0, all memory allocated by #WebKitPermissionStateQuery is
 * released. This function is MT-safe and may be called from any thread.
 *
 * Since: 2.40
 */
void webkit_permission_state_query_unref(WebKitPermissionStateQuery* query)
{
    g_return_if_fail(query);

    if (g_atomic_int_dec_and_test(&query->referenceCount)) {
        query->~WebKitPermissionStateQuery();
        fastFree(query);
    }
}

/**
 * webkit_permission_state_query_get_name:
 * @query: a #WebKitPermissionStateQuery
 *
 * Get the permission name for which access is being queried.
 *
 * Returns: the permission name for @query
 *
 * Since: 2.40
 */
const gchar*
webkit_permission_state_query_get_name(WebKitPermissionStateQuery* query)
{
    g_return_val_if_fail(query, nullptr);

    return query->permissionName.data();
}

/**
 * webkit_permission_state_query_get_security_origin:
 * @query: a #WebKitPermissionStateQuery
 *
 * Get the permission origin for which access is being queried.
 *
 * Returns: (transfer none): A #WebKitSecurityOrigin representing the origin from which the
 * @query was emitted.
 *
 * Since: 2.40
 */
WebKitSecurityOrigin *
webkit_permission_state_query_get_security_origin(WebKitPermissionStateQuery* query)
{
    g_return_val_if_fail(query, nullptr);

    return query->securityOrigin;
}

/**
 * webkit_permission_state_query_finish:
 * @query: a #WebKitPermissionStateQuery
 * @state: a #WebKitPermissionState
 *
 * Notify the web-engine of the selected permission state for the given query. This function should
 * only be called as a response to the `WebKitWebView::query-permission-state` signal.
 *
 * Since: 2.40
 */
void
webkit_permission_state_query_finish(WebKitPermissionStateQuery* query, WebKitPermissionState state)
{
    g_return_if_fail(query);
    g_return_if_fail(query->completionHandler);

    switch (state) {
    case WEBKIT_PERMISSION_STATE_GRANTED:
        query->completionHandler(WebCore::PermissionState::Granted);
        break;
    case WEBKIT_PERMISSION_STATE_DENIED:
        query->completionHandler(WebCore::PermissionState::Denied);
        break;
    case WEBKIT_PERMISSION_STATE_PROMPT:
        query->completionHandler(WebCore::PermissionState::Prompt);
        break;
    }
}
