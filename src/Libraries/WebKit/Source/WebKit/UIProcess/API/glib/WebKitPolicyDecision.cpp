/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "WebKitPolicyDecision.h"

#include "APIWebsitePolicies.h"
#include "WebFramePolicyListenerProxy.h"
#include "WebKitPolicyDecisionPrivate.h"
#include "WebKitWebsitePolicies.h"
#include "WebKitWebsitePoliciesPrivate.h"
#include "WebsitePoliciesData.h"
#include <wtf/glib/WTFGType.h>

using namespace WebKit;

/**
 * WebKitPolicyDecision:
 * @See_also: #WebKitWebView
 *
 * A pending policy decision.
 *
 * Often WebKit allows the client to decide the policy for certain
 * operations. For instance, a client may want to open a link in a new
 * tab, block a navigation entirely, query the user or trigger a download
 * instead of a navigation. In these cases WebKit will fire the
 * #WebKitWebView::decide-policy signal with a #WebKitPolicyDecision
 * object. If the signal handler does nothing, WebKit will act as if
 * webkit_policy_decision_use() was called as soon as signal handling
 * completes. To make a policy decision asynchronously, simply increment
 * the reference count of the #WebKitPolicyDecision object.
 */

struct _WebKitPolicyDecisionPrivate {
    RefPtr<WebFramePolicyListenerProxy> listener;
};

WEBKIT_DEFINE_ABSTRACT_TYPE(WebKitPolicyDecision, webkit_policy_decision, G_TYPE_OBJECT)

static void webkitPolicyDecisionDispose(GObject* object)
{
    webkit_policy_decision_use(WEBKIT_POLICY_DECISION(object));
    G_OBJECT_CLASS(webkit_policy_decision_parent_class)->dispose(object);
}

void webkitPolicyDecisionSetListener(WebKitPolicyDecision* decision, Ref<WebFramePolicyListenerProxy>&& listener)
{
    decision->priv->listener = WTFMove(listener);
}

static void webkit_policy_decision_class_init(WebKitPolicyDecisionClass* decisionClass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(decisionClass);
    objectClass->dispose = webkitPolicyDecisionDispose;
}

/**
 * webkit_policy_decision_use:
 * @decision: a #WebKitPolicyDecision
 *
 * Accept the action which triggered this decision.
 */
void webkit_policy_decision_use(WebKitPolicyDecision* decision)
{
    g_return_if_fail(WEBKIT_IS_POLICY_DECISION(decision));

    if (!decision->priv->listener)
        return;

    auto listener = std::exchange(decision->priv->listener, nullptr);
    listener->use();
}

/**
 * webkit_policy_decision_use_with_policies:
 * @decision: a #WebKitPolicyDecision
 * @policies: a #WebKitWebsitePolicies
 *
 * Accept the navigation action and continue with provided @policies.
 *
 * Accept the navigation action which triggered this decision, and
 * continue with @policies affecting all subsequent loads of resources
 * in the origin associated with the accepted navigation action.
 *
 * For example, a navigation decision to a video sharing website may
 * be accepted under the priviso no movies are allowed to autoplay. The
 * autoplay policy in this case would be set in the @policies.
 *
 * Since: 2.30
 */
void webkit_policy_decision_use_with_policies(WebKitPolicyDecision* decision, WebKitWebsitePolicies* policies)
{
    g_return_if_fail(WEBKIT_IS_POLICY_DECISION(decision));
    g_return_if_fail(WEBKIT_IS_WEBSITE_POLICIES(policies));

    if (!decision->priv->listener)
        return;

    auto listener = std::exchange(decision->priv->listener, nullptr);

    listener->use(webkitWebsitePoliciesGetWebsitePolicies(policies));
}

/**
 * webkit_policy_decision_ignore:
 * @decision: a #WebKitPolicyDecision
 *
 * #WebKitResponsePolicyDecision, this would cancel the request.
 *
 * Ignore the action which triggered this decision. For instance, for a
 * #WebKitResponsePolicyDecision, this would cancel the request.
 */
void webkit_policy_decision_ignore(WebKitPolicyDecision* decision)
{
    g_return_if_fail(WEBKIT_IS_POLICY_DECISION(decision));

    if (!decision->priv->listener)
        return;

    auto listener = std::exchange(decision->priv->listener, nullptr);
    listener->ignore();
}

/**
 * webkit_policy_decision_download:
 * @decision: a #WebKitPolicyDecision
 *
 * Spawn a download from this decision.
 */
void webkit_policy_decision_download(WebKitPolicyDecision* decision)
{
    g_return_if_fail(WEBKIT_IS_POLICY_DECISION(decision));

    if (!decision->priv->listener)
        return;

    auto listener = std::exchange(decision->priv->listener, nullptr);
    listener->download();
}
