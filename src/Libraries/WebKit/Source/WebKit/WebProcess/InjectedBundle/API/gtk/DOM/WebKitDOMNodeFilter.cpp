/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#include "WebKitDOMNodeFilter.h"

#include "GObjectNodeFilterCondition.h"
#include <WebCore/Document.h>
#include <WebCore/NativeNodeFilter.h>
#include "WebKitDOMNode.h"
#include "WebKitDOMNodeFilterPrivate.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

typedef WebKitDOMNodeFilterIface WebKitDOMNodeFilterInterface;

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

G_DEFINE_INTERFACE(WebKitDOMNodeFilter, webkit_dom_node_filter, G_TYPE_OBJECT)

static void webkit_dom_node_filter_default_init(WebKitDOMNodeFilterIface*)
{
}

gshort webkit_dom_node_filter_accept_node(WebKitDOMNodeFilter* filter, WebKitDOMNode* node)
{
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE_FILTER(filter), WEBKIT_DOM_NODE_FILTER_REJECT);
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE(node), WEBKIT_DOM_NODE_FILTER_REJECT);

    return WEBKIT_DOM_NODE_FILTER_GET_IFACE(filter)->accept_node(filter, node);
}

namespace WebKit {

static HashMap<WebCore::NodeFilter*, WebKitDOMNodeFilter*>& nodeFilterMap()
{
    static NeverDestroyed<HashMap<WebCore::NodeFilter*, WebKitDOMNodeFilter*>> nodeFilterMap;
    return nodeFilterMap;
}

static void nodeFilterObjectDestroyedCallback(gpointer coreNodeFilter, GObject* nodeFilter)
{
    WebKitDOMNodeFilter* filter = nodeFilterMap().take(static_cast<WebCore::NodeFilter*>(coreNodeFilter));
    UNUSED_PARAM(nodeFilter);
    ASSERT_UNUSED(filter, reinterpret_cast<GObject*>(filter) == nodeFilter);
}

WebKitDOMNodeFilter* kit(WebCore::NodeFilter* coreNodeFilter)
{
    if (!coreNodeFilter)
        return nullptr;

    return nodeFilterMap().get(coreNodeFilter);
}

RefPtr<WebCore::NodeFilter> core(WebCore::Document* document, WebKitDOMNodeFilter* nodeFilter)
{
    if (!nodeFilter)
        return nullptr;

    RefPtr<WebCore::NodeFilter> coreNodeFilter = static_cast<WebCore::NodeFilter*>(g_object_get_data(G_OBJECT(nodeFilter), "webkit-core-node-filter"));
    if (!coreNodeFilter) {
        coreNodeFilter = WebCore::NativeNodeFilter::create(document, WebKit::GObjectNodeFilterCondition::create(nodeFilter));
        nodeFilterMap().add(coreNodeFilter.get(), nodeFilter);
        g_object_weak_ref(G_OBJECT(nodeFilter), nodeFilterObjectDestroyedCallback, coreNodeFilter.get());
        g_object_set_data(G_OBJECT(nodeFilter), "webkit-core-node-filter", coreNodeFilter.get());
    }
    return coreNodeFilter;
}

} // namespace WebKit
G_GNUC_END_IGNORE_DEPRECATIONS;
