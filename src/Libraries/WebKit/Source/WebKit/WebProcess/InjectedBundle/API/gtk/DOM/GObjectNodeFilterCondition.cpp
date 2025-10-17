/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#include "GObjectNodeFilterCondition.h"

#include "WebKitDOMNodePrivate.h"
#include <WebCore/NodeFilter.h>

using namespace WebCore;

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

GObjectNodeFilterCondition::~GObjectNodeFilterCondition()
{
    g_object_set_data(G_OBJECT(m_filter.get()), "webkit-core-node-filter", nullptr);
}

unsigned short GObjectNodeFilterCondition::acceptNode(Node& node) const
{
    return webkit_dom_node_filter_accept_node(m_filter.get(), WebKit::kit(&node));
}

} // namespace WebKit
G_GNUC_END_IGNORE_DEPRECATIONS;
