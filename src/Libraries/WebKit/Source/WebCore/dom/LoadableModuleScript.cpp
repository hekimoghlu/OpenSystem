/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#include "LoadableModuleScript.h"

#include "Document.h"
#include "Element.h"
#include "LocalFrame.h"
#include "ModuleFetchParameters.h"
#include "ScriptController.h"
#include "ScriptElement.h"

namespace WebCore {

Ref<LoadableModuleScript> LoadableModuleScript::create(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy policy, RequestPriority fetchPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree)
{
    return adoptRef(*new LoadableModuleScript(nonce, integrity, policy, fetchPriority, crossOriginMode, charset, initiatorType, isInUserAgentShadowTree));
}

LoadableModuleScript::LoadableModuleScript(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy policy, RequestPriority fetchPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree)
    : LoadableScript(nonce, policy, fetchPriority, crossOriginMode, charset, initiatorType, isInUserAgentShadowTree)
    , m_parameters(ModuleFetchParameters::create(JSC::ScriptFetchParameters::Type::JavaScript, integrity, /* isTopLevelModule */ true))
{
}

LoadableModuleScript::~LoadableModuleScript() = default;

bool LoadableModuleScript::isLoaded() const
{
    return m_isLoaded;
}

bool LoadableModuleScript::hasError() const
{
    return !!m_error;
}

std::optional<LoadableScript::Error> LoadableModuleScript::takeError()
{
    return std::exchange(m_error, std::nullopt);
}

bool LoadableModuleScript::wasCanceled() const
{
    return m_wasCanceled;
}

void LoadableModuleScript::notifyLoadCompleted(UniquedStringImpl& moduleKey)
{
    m_moduleKey = &moduleKey;
    m_isLoaded = true;
    notifyClientFinished();
}

void LoadableModuleScript::notifyLoadFailed(LoadableScript::Error&& error)
{
    m_error = WTFMove(error);
    m_isLoaded = true;
    notifyClientFinished();
}

void LoadableModuleScript::notifyLoadWasCanceled()
{
    m_wasCanceled = true;
    m_isLoaded = true;
    notifyClientFinished();
}

void LoadableModuleScript::execute(ScriptElement& scriptElement)
{
    scriptElement.executeModuleScript(*this);
}

}
