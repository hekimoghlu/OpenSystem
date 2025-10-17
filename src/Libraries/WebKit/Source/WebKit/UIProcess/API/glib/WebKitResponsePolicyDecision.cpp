/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
#include "WebKitResponsePolicyDecision.h"

#include "WebKitPolicyDecisionPrivate.h"
#include "WebKitResponsePolicyDecisionPrivate.h"
#include "WebKitURIRequestPrivate.h"
#include "WebKitURIResponsePrivate.h"
#include <glib/gi18n-lib.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/WTFGType.h>
#include <wtf/text/CString.h>

using namespace WebKit;
using namespace WebCore;

/**
 * WebKitResponsePolicyDecision:
 * @See_also: #WebKitPolicyDecision, #WebKitWebView
 *
 * A policy decision for resource responses.
 *
 * WebKitResponsePolicyDecision represents a policy decision for a
 * resource response, whether from the network or the local system.
 * A very common use case for these types of decision is deciding
 * whether or not to download a particular resource or to load it
 * normally.
 */

struct _WebKitResponsePolicyDecisionPrivate {
    RefPtr<API::NavigationResponse> navigationResponse;
    GRefPtr<WebKitURIRequest> request;
    GRefPtr<WebKitURIResponse> response;
};

WEBKIT_DEFINE_FINAL_TYPE(WebKitResponsePolicyDecision, webkit_response_policy_decision, WEBKIT_TYPE_POLICY_DECISION, WebKitPolicyDecision)

enum {
    PROP_0,
    PROP_REQUEST,
    PROP_RESPONSE,
};

static void webkitResponsePolicyDecisionGetProperty(GObject* object, guint propId, GValue* value, GParamSpec* paramSpec)
{
    WebKitResponsePolicyDecision* decision = WEBKIT_RESPONSE_POLICY_DECISION(object);
    switch (propId) {
    case PROP_REQUEST:
        g_value_set_object(value, webkit_response_policy_decision_get_request(decision));
        break;
    case PROP_RESPONSE:
        g_value_set_object(value, webkit_response_policy_decision_get_response(decision));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propId, paramSpec);
        break;
    }
}

static void webkit_response_policy_decision_class_init(WebKitResponsePolicyDecisionClass* decisionClass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(decisionClass);
    objectClass->get_property = webkitResponsePolicyDecisionGetProperty;

    /**
     * WebKitResponsePolicyDecision:request:
     *
     * This property contains the #WebKitURIRequest associated with this
     * policy decision.
     */
    g_object_class_install_property(objectClass,
        PROP_REQUEST,
        g_param_spec_object("request",
            nullptr, nullptr,
            WEBKIT_TYPE_URI_REQUEST,
            WEBKIT_PARAM_READABLE));

    /**
     * WebKitResponsePolicyDecision:response:
     *
     * This property contains the #WebKitURIResponse associated with this
     * policy decision.
     */
    g_object_class_install_property(objectClass,
        PROP_RESPONSE,
        g_param_spec_object("response",
            nullptr, nullptr,
            WEBKIT_TYPE_URI_RESPONSE,
            WEBKIT_PARAM_READABLE));

}

/**
 * webkit_response_policy_decision_get_request:
 * @decision: a #WebKitResponsePolicyDecision
 *
 * Return the #WebKitURIRequest associated with the response decision.
 *
 * Modifications to the returned object are <emphasis>not</emphasis> taken
 * into account when the request is sent over the network, and is intended
 * only to aid in evaluating whether a response decision should be taken or
 * not. To modify requests before they are sent over the network the
 * #WebKitPage::send-request signal can be used instead.
 *
 * Returns: (transfer none): The URI request that is associated with this policy decision.
 */
WebKitURIRequest* webkit_response_policy_decision_get_request(WebKitResponsePolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_RESPONSE_POLICY_DECISION(decision), nullptr);
    if (!decision->priv->request)
        decision->priv->request = adoptGRef(webkitURIRequestCreateForResourceRequest(decision->priv->navigationResponse->request()));
    return decision->priv->request.get();
}

/**
 * webkit_response_policy_decision_get_response:
 * @decision: a #WebKitResponsePolicyDecision
 *
 * Gets the value of the #WebKitResponsePolicyDecision:response property.
 *
 * Returns: (transfer none): The URI response that is associated with this policy decision.
 */
WebKitURIResponse* webkit_response_policy_decision_get_response(WebKitResponsePolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_RESPONSE_POLICY_DECISION(decision), nullptr);
    if (!decision->priv->response)
        decision->priv->response = adoptGRef(webkitURIResponseCreateForResourceResponse(decision->priv->navigationResponse->response()));
    return decision->priv->response.get();
}

/**
 * webkit_response_policy_decision_is_mime_type_supported:
 * @decision: a #WebKitResponsePolicyDecision
 *
 * Gets whether the MIME type of the response can be displayed in the #WebKitWebView.
 *
 * Gets whether the MIME type of the response can be displayed in the #WebKitWebView
 * that triggered this policy decision request. See also webkit_web_view_can_show_mime_type().
 *
 * Returns: %TRUE if the MIME type of the response is supported or %FALSE otherwise
 *
 * Since: 2.4
 */
gboolean webkit_response_policy_decision_is_mime_type_supported(WebKitResponsePolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_RESPONSE_POLICY_DECISION(decision), FALSE);
    return decision->priv->navigationResponse->canShowMIMEType();
}

/**
 * webkit_response_policy_decision_is_main_frame_main_resource:
 * @decision: a #WebKitResponsePolicyDecision
 *
 * Gets whether the request is the main frame main resource
 *
 * Returns: %TRUE if the request is the main frame main resouce or %FALSE otherwise
 *
 * Since: 2.40
 */
gboolean webkit_response_policy_decision_is_main_frame_main_resource(WebKitResponsePolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_RESPONSE_POLICY_DECISION(decision), FALSE);

    if (!decision->priv->navigationResponse->frame().isMainFrame())
        return FALSE;

    return decision->priv->navigationResponse->request().requester() == ResourceRequestRequester::Main;
}

WebKitPolicyDecision* webkitResponsePolicyDecisionCreate(Ref<API::NavigationResponse>&& response, Ref<WebKit::WebFramePolicyListenerProxy>&& listener)
{
    WebKitResponsePolicyDecision* responseDecision = WEBKIT_RESPONSE_POLICY_DECISION(g_object_new(WEBKIT_TYPE_RESPONSE_POLICY_DECISION, nullptr));
    responseDecision->priv->navigationResponse = WTFMove(response);
    WebKitPolicyDecision* decision = WEBKIT_POLICY_DECISION(responseDecision);
    webkitPolicyDecisionSetListener(decision, WTFMove(listener));
    return decision;
}
