/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
#include "WorkerLocation.h"

#include "SecurityOrigin.h"
#include "WebCoreOpaqueRoot.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

String WorkerLocation::href() const
{
    return m_url.string();
}

String WorkerLocation::protocol() const
{
    return makeString(m_url.protocol(), ':');
}

String WorkerLocation::host() const
{
    return m_url.hostAndPort();
}

String WorkerLocation::hostname() const
{
    return m_url.host().toString();
}

String WorkerLocation::port() const
{
    auto port = m_url.port();
    return port ? String::number(*port) : emptyString();
}

String WorkerLocation::pathname() const
{
    auto path = m_url.path();
    return path.isEmpty() ? "/"_s : path.toString();
}

String WorkerLocation::search() const
{
    return m_url.query().isEmpty() ? emptyString() : m_url.queryWithLeadingQuestionMark().toString();
}

String WorkerLocation::hash() const
{
    return m_url.fragmentIdentifier().isEmpty() ? emptyString() : m_url.fragmentIdentifierWithLeadingNumberSign().toString();
}

String WorkerLocation::origin() const
{
    return m_origin;
}

WebCoreOpaqueRoot root(WorkerLocation* location)
{
    return WebCoreOpaqueRoot { location };
}

} // namespace WebCore
