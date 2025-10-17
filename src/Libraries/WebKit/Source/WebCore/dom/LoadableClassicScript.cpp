/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#include "LoadableClassicScript.h"

#include "DefaultResourceLoadPriority.h"
#include "Element.h"
#include "FetchIdioms.h"
#include "LoadableScriptError.h"
#include "ScriptElement.h"
#include "ScriptSourceCode.h"
#include "SubresourceIntegrity.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringImpl.h>

namespace WebCore {

LoadableNonModuleScriptBase::LoadableNonModuleScriptBase(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy policy, RequestPriority fetchPriority,  const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree, bool isAsync)
    : LoadableScript(nonce, policy, fetchPriority, crossOriginMode, charset, initiatorType, isInUserAgentShadowTree)
    , m_integrity(integrity)
    , m_isAsync(isAsync)
{
}

LoadableNonModuleScriptBase::~LoadableNonModuleScriptBase()
{
    if (CachedResourceHandle cachedScript = m_cachedScript)
        cachedScript->removeClient(*this);
}

bool LoadableNonModuleScriptBase::isLoaded() const
{
    ASSERT(m_cachedScript);
    return m_cachedScript->isLoaded();
}

bool LoadableNonModuleScriptBase::hasError() const
{
    ASSERT(m_cachedScript);
    if (m_error)
        return true;

    return m_cachedScript->errorOccurred();
}

std::optional<LoadableScript::Error> LoadableNonModuleScriptBase::takeError()
{
    ASSERT(m_cachedScript);
    if (m_error)
        return std::exchange(m_error, { });

    if (m_cachedScript->errorOccurred())
        return Error { ErrorType::Fetch, { }, { } };

    return std::nullopt;
}

bool LoadableNonModuleScriptBase::wasCanceled() const
{
    ASSERT(m_cachedScript);
    return m_cachedScript->wasCanceled();
}

void LoadableNonModuleScriptBase::notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    ASSERT(m_cachedScript);
    if (resource.resourceError().isAccessControl()) {
        static NeverDestroyed<String> consoleMessage(MAKE_STATIC_STRING_IMPL("Cross-origin script load denied by Cross-Origin Resource Sharing policy."));
        m_error = Error {
            ErrorType::CrossOriginLoad,
            ConsoleMessage {
                MessageSource::JS,
                MessageLevel::Error,
                consoleMessage
            },
            { }
        };
    }

    CachedResourceHandle cachedScript = m_cachedScript;
    if (!m_error && !isScriptAllowedByNosniff(cachedScript->response())) {
        m_error = Error {
            ErrorType::Nosniff,
            ConsoleMessage {
                MessageSource::Security,
                MessageLevel::Error,
                makeString("Refused to execute "_s, cachedScript->url().stringCenterEllipsizedToLength(), " as script because \"X-Content-Type-Options: nosniff\" was given and its Content-Type is not a script MIME type."_s)
            },
            { }
        };
    }

    if (!m_error && shouldBlockResponseDueToMIMEType(cachedScript->response(), cachedScript->options().destination)) {
        m_error = Error {
            ErrorType::MIMEType,
            ConsoleMessage {
                MessageSource::Security,
                MessageLevel::Error,
                makeString("Refused to execute "_s, cachedScript->url().stringCenterEllipsizedToLength(), " as script because "_s, cachedScript->response().mimeType(), " is not a script MIME type."_s)
            },
            { }
        };
    }

    if (!m_error && !resource.errorOccurred() && !matchIntegrityMetadata(resource, m_integrity)) {
        m_error = Error {
            ErrorType::FailedIntegrityCheck,
            ConsoleMessage { MessageSource::Security, MessageLevel::Error, makeString("Cannot load script "_s, integrityMismatchDescription(resource, m_integrity)) },
            { }
        };
    }

    notifyClientFinished();
}

bool LoadableNonModuleScriptBase::load(Document& document, const URL& sourceURL)
{
    ASSERT(!m_cachedScript);

    auto priority = [&]() -> std::optional<ResourceLoadPriority> {
        if (isAsync())
            return DefaultResourceLoadPriority::asyncScript;
        // Use default.
        return { };
    };

    m_weakDocument = document;
    CachedResourceHandle cachedScript = requestScriptWithCache(document, sourceURL, crossOriginMode(), String { integrity() }, priority(), { });
    m_cachedScript = cachedScript;
    if (!cachedScript)
        return false;
    cachedScript->addClient(*this);
    return true;
}

Ref<LoadableClassicScript> LoadableClassicScript::create(const AtomString& nonce, const AtomString& integrityMetadata, ReferrerPolicy policy, RequestPriority fetchPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree, bool isAsync)
{
    return adoptRef(*new LoadableClassicScript(nonce, integrityMetadata, policy, fetchPriority, crossOriginMode, charset, initiatorType, isInUserAgentShadowTree, isAsync));
}

LoadableClassicScript::LoadableClassicScript(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy policy, RequestPriority fetchPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree, bool isAsync)
    : LoadableNonModuleScriptBase(nonce, integrity, policy, fetchPriority, crossOriginMode, charset, initiatorType, isInUserAgentShadowTree, isAsync)
{
}

void LoadableClassicScript::execute(ScriptElement& scriptElement)
{
    ASSERT(!m_error);
    scriptElement.executeClassicScript(ScriptSourceCode(protectedCachedScript().get(), JSC::SourceProviderSourceType::Program, *this));
}

}
