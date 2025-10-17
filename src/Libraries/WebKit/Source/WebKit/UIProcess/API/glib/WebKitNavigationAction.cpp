/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#include "WebKitNavigationAction.h"

#include "WebKitNavigationActionPrivate.h"
#include "WebKitPrivate.h"
#include "WebKitURIRequestPrivate.h"

using namespace WebKit;

/**
 * WebKitNavigationAction:
 *
 * Provides details about interaction resulting in a resource load.
 */

G_DEFINE_BOXED_TYPE(WebKitNavigationAction, webkit_navigation_action, webkit_navigation_action_copy, webkit_navigation_action_free)

WebKitNavigationAction* webkitNavigationActionCreate(Ref<API::NavigationAction>&& action)
{
    WebKitNavigationAction* navigation = static_cast<WebKitNavigationAction*>(fastZeroedMalloc(sizeof(WebKitNavigationAction)));
    new (navigation) WebKitNavigationAction(WTFMove(action));
    return navigation;
}

/**
 * webkit_navigation_action_copy:
 * @navigation: a #WebKitNavigationAction
 *
 * Make a copy of @navigation.
 *
 * Returns: (transfer full): A copy of passed in #WebKitNavigationAction
 *
 * Since: 2.6
 */
WebKitNavigationAction* webkit_navigation_action_copy(WebKitNavigationAction* navigation)
{
    g_return_val_if_fail(navigation, nullptr);

    WebKitNavigationAction* copy = static_cast<WebKitNavigationAction*>(fastZeroedMalloc(sizeof(WebKitNavigationAction)));
    new (copy) WebKitNavigationAction(navigation);
    return copy;
}

/**
 * webkit_navigation_action_free:
 * @navigation: a #WebKitNavigationAction
 *
 * Free the #WebKitNavigationAction
 *
 * Since: 2.6
 */
void webkit_navigation_action_free(WebKitNavigationAction* navigation)
{
    g_return_if_fail(navigation);

    navigation->~WebKitNavigationAction();
    fastFree(navigation);
}

/**
 * webkit_navigation_action_get_navigation_type:
 * @navigation: a #WebKitNavigationAction
 *
 * Return the type of action that triggered the navigation.
 *
 * Returns: a #WebKitNavigationType
 *
 * Since: 2.6
 */
WebKitNavigationType webkit_navigation_action_get_navigation_type(WebKitNavigationAction* navigation)
{
    g_return_val_if_fail(navigation, WEBKIT_NAVIGATION_TYPE_OTHER);
    return toWebKitNavigationType(navigation->action->navigationType());
}

/**
 * webkit_navigation_action_get_mouse_button:
 * @navigation: a #WebKitNavigationAction
 *
 * Return the number of the mouse button that triggered the navigation.
 *
 * Return the number of the mouse button that triggered the navigation, or 0 if
 * the navigation was not started by a mouse event.
 *
 * Returns: the mouse button number or 0
 *
 * Since: 2.6
 */
unsigned webkit_navigation_action_get_mouse_button(WebKitNavigationAction* navigation)
{
    g_return_val_if_fail(navigation, 0);
    return toWebKitMouseButton(navigation->action->mouseButton());
}

/**
 * webkit_navigation_action_get_modifiers:
 * @navigation: a #WebKitNavigationAction
 *
 * Return the modifier keys.
 *
 * Return a bitmask of #GdkModifierType values describing the modifier keys that were in effect
 * when the navigation was requested
 *
 * Returns: the modifier keys
 *
 * Since: 2.6
 */
unsigned webkit_navigation_action_get_modifiers(WebKitNavigationAction* navigation)
{
    g_return_val_if_fail(navigation, 0);
    return toPlatformModifiers(navigation->action->modifiers());
}

/**
 * webkit_navigation_action_get_request:
 * @navigation: a #WebKitNavigationAction
 *
 * Return the #WebKitURIRequest associated with the navigation action.
 *
 * Modifications to the returned object are <emphasis>not</emphasis> taken
 * into account when the request is sent over the network, and is intended
 * only to aid in evaluating whether a navigation action should be taken or
 * not. To modify requests before they are sent over the network the
 * #WebKitPage::send-request signal can be used instead.
 *
 * Returns: (transfer none): a #WebKitURIRequest
 *
 * Since: 2.6
 */
WebKitURIRequest* webkit_navigation_action_get_request(WebKitNavigationAction* navigation)
{
    g_return_val_if_fail(navigation, nullptr);
    if (!navigation->request)
        navigation->request = adoptGRef(webkitURIRequestCreateForResourceRequest(navigation->action->request()));
    return navigation->request.get();
}

/**
 * webkit_navigation_action_is_user_gesture:
 * @navigation: a #WebKitNavigationAction
 *
 * Return whether the navigation was triggered by a user gesture like a mouse click.
 *
 * Returns: whether navigation action is a user gesture
 *
 * Since: 2.6
 */
gboolean webkit_navigation_action_is_user_gesture(WebKitNavigationAction* navigation)
{
    g_return_val_if_fail(navigation, FALSE);
    return navigation->action->isProcessingUserGesture();
}

/**
 * webkit_navigation_action_is_redirect:
 * @navigation: a #WebKitNavigationAction
 *
 * Returns whether the @navigation was redirected.
 *
 * Returns: %TRUE if the original navigation was redirected, %FALSE otherwise.
 *
 * Since: 2.20
 */
gboolean webkit_navigation_action_is_redirect(WebKitNavigationAction* navigation)
{
    g_return_val_if_fail(navigation, FALSE);
    return navigation->action->isRedirect();
}

/**
 * webkit_navigation_action_get_frame_name:
 * @navigation: a #WebKitNavigationAction
 *
 * Gets the @navigation target frame name. For example if navigation was triggered by clicking a
 * link with a target attribute equal to "_blank", this will return the value of that attribute.
 * In all other cases this function will return %NULL.
 *
 * Returns: (nullable): The name of the new frame this navigation action targets or %NULL
 *
 * Since: 2.40
 */
const char* webkit_navigation_action_get_frame_name(WebKitNavigationAction* navigation)
{
    g_return_val_if_fail(navigation, nullptr);
    if (!navigation->frameName) {
        if (auto targetFrameName = navigation->action->targetFrameName(); !!targetFrameName)
            navigation->frameName = targetFrameName.utf8();
        else
            navigation->frameName = CString();
    }
    return navigation->frameName->data();
}
