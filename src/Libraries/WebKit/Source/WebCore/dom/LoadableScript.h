/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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
#pragma once

#include "ScriptElementCachedScriptFetcher.h"
#include <wtf/CheckedRef.h>
#include <wtf/WeakHashCountedSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class LoadableScriptClient;
class ScriptElement;

struct LoadableScriptConsoleMessage;
struct LoadableScriptError;

enum class LoadableScriptErrorType : uint8_t;

class LoadableScript : public ScriptElementCachedScriptFetcher {
public:
    using ConsoleMessage = LoadableScriptConsoleMessage;
    using Error = LoadableScriptError;
    using ErrorType = LoadableScriptErrorType;

    virtual ~LoadableScript();

    virtual bool isLoaded() const = 0;
    virtual bool hasError() const = 0;
    virtual std::optional<Error> takeError() = 0;
    virtual bool wasCanceled() const = 0;

    virtual void execute(ScriptElement&) = 0;

    void addClient(LoadableScriptClient&);
    void removeClient(LoadableScriptClient&);

protected:
    LoadableScript(const AtomString& nonce, ReferrerPolicy, RequestPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree);

    void notifyClientFinished();

private:
    WeakHashCountedSet<LoadableScriptClient> m_clients;
};

}
