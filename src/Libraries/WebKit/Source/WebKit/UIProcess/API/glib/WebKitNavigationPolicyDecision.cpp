/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
#include "WebKitNavigationPolicyDecision.h"

#include "WebKitEnumTypes.h"
#include "WebKitNavigationActionPrivate.h"
#include "WebKitNavigationPolicyDecisionPrivate.h"
#include "WebKitPolicyDecisionPrivate.h"
#include "WebKitURIRequestPrivate.h"
#include <glib/gi18n-lib.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/WTFGType.h>

using namespace WebKit;
using namespace WebCore;

/**
 * WebKitNavigationPolicyDecision:
 * @See_also: #WebKitPolicyDecision, #WebKitWebView
 *
 * A policy decision for navigation actions.
 *
 * WebKitNavigationPolicyDecision represents a policy decision for events associated with
 * navigations. If the value of #WebKitNavigationPolicyDecision:mouse-button is not 0, then
 * the navigation was triggered by a mouse event.
 */

struct _WebKitNavigationPolicyDecisionPrivate {
    ~_WebKitNavigationPolicyDecisionPrivate()
    {
        webkit_navigation_action_free(navigationAction);
    }

    WebKitNavigationAction* navigationAction;
};

WEBKIT_DEFINE_FINAL_TYPE(WebKitNavigationPolicyDecision, webkit_navigation_policy_decision, WEBKIT_TYPE_POLICY_DECISION, WebKitPolicyDecision)

enum {
    PROP_0,
    PROP_NAVIGATION_ACTION,
#if PLATFORM(GTK) && !USE(GTK4)
    PROP_NAVIGATION_TYPE,
    PROP_MOUSE_BUTTON,
    PROP_MODIFIERS,
    PROP_REQUEST,
#endif
#if !ENABLE(2022_GLIB_API)
    PROP_FRAME_NAME,
#endif
};

static void webkitNavigationPolicyDecisionGetProperty(GObject* object, guint propId, GValue* value, GParamSpec* paramSpec)
{
    WebKitNavigationPolicyDecision* decision = WEBKIT_NAVIGATION_POLICY_DECISION(object);
    switch (propId) {
    case PROP_NAVIGATION_ACTION:
        g_value_set_boxed(value, webkit_navigation_policy_decision_get_navigation_action(decision));
        break;
#if PLATFORM(GTK) && !USE(GTK4)
    case PROP_NAVIGATION_TYPE:
        g_value_set_enum(value, webkit_navigation_action_get_navigation_type(decision->priv->navigationAction));
        break;
    case PROP_MOUSE_BUTTON:
        g_value_set_enum(value, webkit_navigation_action_get_mouse_button(decision->priv->navigationAction));
        break;
    case PROP_MODIFIERS:
        g_value_set_uint(value, webkit_navigation_action_get_modifiers(decision->priv->navigationAction));
        break;
    case PROP_REQUEST:
        g_value_set_object(value, webkit_navigation_action_get_request(decision->priv->navigationAction));
        break;
#endif
#if !ENABLE(2022_GLIB_API)
    case PROP_FRAME_NAME:
        g_value_set_string(value, webkit_navigation_action_get_frame_name(decision->priv->navigationAction));
        break;
#endif
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propId, paramSpec);
        break;
    }
}

static void webkit_navigation_policy_decision_class_init(WebKitNavigationPolicyDecisionClass* decisionClass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(decisionClass);
    objectClass->get_property = webkitNavigationPolicyDecisionGetProperty;

    /**
     * WebKitNavigationPolicyDecision:navigation-action:
     *
     * The #WebKitNavigationAction that triggered this policy decision.
     *
     * Since: 2.6
     */
    g_object_class_install_property(
        objectClass,
        PROP_NAVIGATION_ACTION,
        g_param_spec_boxed(
            "navigation-action",
            nullptr, nullptr,
            WEBKIT_TYPE_NAVIGATION_ACTION,
            WEBKIT_PARAM_READABLE));

#if PLATFORM(GTK) && !USE(GTK4)
    /**
     * WebKitNavigationPolicyDecision:navigation-type:
     *
     * The type of navigation that triggered this policy decision. This is
     * useful for enacting different policies depending on what type of user
     * action caused the navigation.
     *
     * Deprecated: 2.6: Use #WebKitNavigationPolicyDecision:navigation-action instead
     */
    g_object_class_install_property(objectClass,
                                    PROP_NAVIGATION_TYPE,
                                    g_param_spec_enum("navigation-type",
                                                      nullptr, nullptr,
                                                      WEBKIT_TYPE_NAVIGATION_TYPE,
                                                      WEBKIT_NAVIGATION_TYPE_LINK_CLICKED,
                                                      WEBKIT_PARAM_READABLE));

    /**
     * WebKitNavigationPolicyDecision:mouse-button:
     *
     * If the navigation associated with this policy decision was originally
     * triggered by a mouse event, this property contains non-zero button number
     * of the button triggering that event. The button numbers match those from GDK.
     * If the navigation was not triggered by a mouse event, the value of this
     * property will be 0.
     *
     * Deprecated: 2.6: Use #WebKitNavigationPolicyDecision:navigation-action instead
     */
    g_object_class_install_property(objectClass,
                                    PROP_MOUSE_BUTTON,
                                    g_param_spec_uint("mouse-button",
                                                      nullptr, nullptr,
                                                      0, G_MAXUINT, 0,
                                                      WEBKIT_PARAM_READABLE));

    /**
     * WebKitNavigationPolicyDecision:modifiers:
     *
     * If the navigation associated with this policy decision was originally
     * triggered by a mouse event, this property contains a bitmask of various
     * #GdkModifierType values describing the modifiers used for that click.
     * If the navigation was not triggered by a mouse event or no modifiers
     * were active, the value of this property will be zero.
     *
     * Deprecated: 2.6: Use #WebKitNavigationPolicyDecision:navigation-action instead
     */
    g_object_class_install_property(objectClass,
                                    PROP_MODIFIERS,
                                    g_param_spec_uint("modifiers",
                                                      nullptr, nullptr,
                                                      0, G_MAXUINT, 0,
                                                      WEBKIT_PARAM_READABLE));

    /**
     * WebKitNavigationPolicyDecision:request:
     *
     * This property contains the #WebKitURIRequest associated with this
     * navigation.
     *
     * Deprecated: 2.6: Use #WebKitNavigationPolicyDecision:navigation-action instead
     */
    g_object_class_install_property(objectClass,
                                    PROP_REQUEST,
                                    g_param_spec_object("request",
                                                      nullptr, nullptr,
                                                      WEBKIT_TYPE_URI_REQUEST,
                                                      WEBKIT_PARAM_READABLE));
#endif

#if !ENABLE(2022_GLIB_API)
    /**
     * WebKitNavigationPolicyDecision:frame-name:
     *
     * If this navigation request targets a new frame, this property contains
     * the name of that frame. For example if the decision was triggered by clicking a
     * link with a target attribute equal to "_blank", this property will contain the
     * value of that attribute. In all other cases, this value will be %NULL.
     *
     * Deprecated: 2.40: Use #WebKitNavigationPolicyDecision:navigation-action instead
     */
    g_object_class_install_property(objectClass,
                                    PROP_FRAME_NAME,
                                    g_param_spec_string("frame-name",
                                                      nullptr, nullptr,
                                                      0,
                                                      WEBKIT_PARAM_READABLE));
#endif
}

/**
 * webkit_navigation_policy_decision_get_navigation_action:
 * @decision: a #WebKitNavigationPolicyDecision
 *
 * Gets the value of the #WebKitNavigationPolicyDecision:navigation-action property.
 *
 * Returns: (transfer none): The #WebKitNavigationAction triggering this policy decision.
 *
 * Since: 2.6
 */
WebKitNavigationAction* webkit_navigation_policy_decision_get_navigation_action(WebKitNavigationPolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_NAVIGATION_POLICY_DECISION(decision), nullptr);
    return decision->priv->navigationAction;
}

#if PLATFORM(GTK) && !USE(GTK4)
/**
 * webkit_navigation_policy_decision_get_navigation_type:
 * @decision: a #WebKitNavigationPolicyDecision
 *
 * Gets the value of the #WebKitNavigationPolicyDecision:navigation-type property.
 *
 * Returns: The type of navigation triggering this policy decision.
 *
 * Deprecated: 2.6: Use webkit_navigation_policy_decision_get_navigation_action() instead.
 */
WebKitNavigationType webkit_navigation_policy_decision_get_navigation_type(WebKitNavigationPolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_NAVIGATION_POLICY_DECISION(decision), WEBKIT_NAVIGATION_TYPE_OTHER);
    return webkit_navigation_action_get_navigation_type(decision->priv->navigationAction);
}

/**
 * webkit_navigation_policy_decision_get_mouse_button:
 * @decision: a #WebKitNavigationPolicyDecision
 *
 * Gets the value of the #WebKitNavigationPolicyDecision:mouse-button property.
 *
 * Returns: The mouse button used if this decision was triggered by a mouse event or 0 otherwise
 *
 * Deprecated: 2.6: Use webkit_navigation_policy_decision_get_navigation_action() instead.
 */
guint webkit_navigation_policy_decision_get_mouse_button(WebKitNavigationPolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_NAVIGATION_POLICY_DECISION(decision), 0);
    return webkit_navigation_action_get_mouse_button(decision->priv->navigationAction);
}

/**
 * webkit_navigation_policy_decision_get_modifiers:
 * @decision: a #WebKitNavigationPolicyDecision
 *
 * Gets the value of the #WebKitNavigationPolicyDecision:modifiers property.
 *
 * Returns: The modifiers active if this decision was triggered by a mouse event
 *
 * Deprecated: 2.6: Use webkit_navigation_policy_decision_get_navigation_action() instead.
 */
unsigned webkit_navigation_policy_decision_get_modifiers(WebKitNavigationPolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_NAVIGATION_POLICY_DECISION(decision), 0);
    return webkit_navigation_action_get_modifiers(decision->priv->navigationAction);
}

/**
 * webkit_navigation_policy_decision_get_request:
 * @decision: a #WebKitNavigationPolicyDecision
 *
 * Gets the value of the #WebKitNavigationPolicyDecision:request property.
 *
 * Returns: (transfer none): The URI request that is associated with this navigation
 *
 * Deprecated: 2.6: Use webkit_navigation_policy_decision_get_navigation_action() instead.
 */
WebKitURIRequest* webkit_navigation_policy_decision_get_request(WebKitNavigationPolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_NAVIGATION_POLICY_DECISION(decision), nullptr);
    return webkit_navigation_action_get_request(decision->priv->navigationAction);
}
#endif

#if !ENABLE(2022_GLIB_API)
/**
 * webkit_navigation_policy_decision_get_frame_name:
 * @decision: a #WebKitNavigationPolicyDecision
 *
 * Gets the value of the #WebKitNavigationPolicyDecision:frame-name property.
 *
 * Returns: The name of the new frame this navigation action targets or %NULL
 *
 * Deprecated: 2.40: Use webkit_navigation_policy_decision_get_navigation_action() instead.
 */
const char* webkit_navigation_policy_decision_get_frame_name(WebKitNavigationPolicyDecision* decision)
{
    g_return_val_if_fail(WEBKIT_IS_NAVIGATION_POLICY_DECISION(decision), nullptr);
    return webkit_navigation_action_get_frame_name(decision->priv->navigationAction);
}
#endif

WebKitPolicyDecision* webkitNavigationPolicyDecisionCreate(Ref<API::NavigationAction>&& navigationAction, Ref<WebFramePolicyListenerProxy>&& listener)
{
    WebKitNavigationPolicyDecision* navigationDecision = WEBKIT_NAVIGATION_POLICY_DECISION(g_object_new(WEBKIT_TYPE_NAVIGATION_POLICY_DECISION, nullptr));
    navigationDecision->priv->navigationAction = webkitNavigationActionCreate(WTFMove(navigationAction));
    WebKitPolicyDecision* decision = WEBKIT_POLICY_DECISION(navigationDecision);
    webkitPolicyDecisionSetListener(decision, WTFMove(listener));
    return decision;
}
