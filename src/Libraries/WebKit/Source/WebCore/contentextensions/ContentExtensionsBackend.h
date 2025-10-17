/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "CompiledContentExtension.h"
#include "ContentExtension.h"
#include "ContentExtensionRule.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class DocumentLoader;
class Page;
class ResourceRequest;
struct ContentRuleListResults;

namespace ContentExtensions {

struct ResourceLoadInfo;

// The ContentExtensionsBackend is the internal model of all the content extensions.
//
// It provides two services:
// 1) It stores the rules for each content extension.
// 2) It provides APIs for the WebCore interfaces to use those rules efficiently.
class ContentExtensionsBackend {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ContentExtensionsBackend, WEBCORE_EXPORT);
public:
    // - Rule management interface. This can be used by upper layer.

    // Set a list of rules for a given name. If there were existing rules for the name, they are overridden.
    // The identifier cannot be empty.
    WEBCORE_EXPORT void addContentExtension(const String& identifier, Ref<CompiledContentExtension>, URL&& extensionBaseURL, ContentExtension::ShouldCompileCSS = ContentExtension::ShouldCompileCSS::Yes);
    WEBCORE_EXPORT void removeContentExtension(const String& identifier);
    WEBCORE_EXPORT void removeAllContentExtensions();

    // - Internal WebCore Interface.
    struct ActionsFromContentRuleList {
        String contentRuleListIdentifier;
        bool sawIgnorePreviousRules { false };
        Vector<DeserializedAction> actions;
    };

    enum class ShouldSkipRuleList : bool { No, Yes };
    using RuleListFilter = Function<ShouldSkipRuleList(const String&)>;
    WEBCORE_EXPORT Vector<ActionsFromContentRuleList> actionsForResourceLoad(const ResourceLoadInfo&, const RuleListFilter& = { [](const String&) { return ShouldSkipRuleList::No; } }) const;
    WEBCORE_EXPORT StyleSheetContents* globalDisplayNoneStyleSheet(const String& identifier) const;

    ContentRuleListResults processContentRuleListsForLoad(Page&, const URL&, OptionSet<ResourceType>, DocumentLoader& initiatingDocumentLoader, const URL& redirectFrom, const RuleListFilter&);
    WEBCORE_EXPORT ContentRuleListResults processContentRuleListsForPingLoad(const URL&, const URL& mainDocumentURL, const URL& frameURL);
    bool processContentRuleListsForResourceMonitoring(const URL&, const URL& mainDocumentURL, const URL& frameURL, OptionSet<ResourceType>);

    static const String& displayNoneCSSRule();

    void forEach(const Function<void(const String&, ContentExtension&)>&);

    WEBCORE_EXPORT static bool shouldBeMadeSecure(const URL&);

    ContentExtensionsBackend() = default;
    ContentExtensionsBackend isolatedCopy() && { return ContentExtensionsBackend { crossThreadCopy(WTFMove(m_contentExtensions)) }; }

private:
    explicit ContentExtensionsBackend(UncheckedKeyHashMap<String, Ref<ContentExtension>>&& contentExtensions)
        : m_contentExtensions(WTFMove(contentExtensions))
    {
    }

    ActionsFromContentRuleList actionsFromContentRuleList(const ContentExtension&, const String& urlString, const ResourceLoadInfo&, ResourceFlags) const;

    UncheckedKeyHashMap<String, Ref<ContentExtension>> m_contentExtensions;
};

WEBCORE_EXPORT void applyResultsToRequest(ContentRuleListResults&&, Page*, ResourceRequest&);
std::optional<String> customTrackerBlockingMessageForConsole(const ContentRuleListResults&, const URL& urlString = { }, const URL& mainDocumentURL = { });

} // namespace WebCore::ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
