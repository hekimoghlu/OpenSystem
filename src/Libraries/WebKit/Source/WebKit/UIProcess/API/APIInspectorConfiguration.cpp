/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "APIInspectorConfiguration.h"

#include "WebProcessPool.h"
#include "WebURLSchemeHandler.h"

namespace API {

Ref<InspectorConfiguration> InspectorConfiguration::create()
{
    return adoptRef(*new InspectorConfiguration);
}

InspectorConfiguration::~InspectorConfiguration() = default;

void InspectorConfiguration::addURLSchemeHandler(Ref<WebKit::WebURLSchemeHandler>&& urlSchemeHandler, const WTF::String& urlScheme)
{
    m_customURLSchemes.append(std::make_pair(WTFMove(urlSchemeHandler), urlScheme));
}

WebKit::WebProcessPool* InspectorConfiguration::processPool()
{
    return m_processPool.get();
}

void InspectorConfiguration::setProcessPool(WebKit::WebProcessPool* processPool)
{
    m_processPool = processPool;
}

} // namespace API

