/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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

#include "APIObject.h"
#include <system_error>
#include <wtf/text/WTFString.h>

namespace WebCore {
class FragmentedSharedBuffer;
}

namespace WTF {
class ConcurrentWorkQueue;
class WorkQueue;
}

namespace API {

class ContentRuleList;

class ContentRuleListStore final : public ObjectImpl<Object::Type::ContentRuleListStore> {
public:
    enum class Error {
        LookupFailed = 1,
        VersionMismatch,
        CompileFailed,
        RemoveFailed
    };

#if ENABLE(CONTENT_EXTENSIONS)
    // This should be incremented every time a functional change is made to the bytecode, file format, etc.
    // to prevent crashing while loading old data.
    static constexpr uint32_t CurrentContentRuleListFileVersion = 18;

    static ContentRuleListStore& defaultStoreSingleton();
    static Ref<ContentRuleListStore> storeWithPath(const WTF::String& storePath);

    explicit ContentRuleListStore();
    explicit ContentRuleListStore(const WTF::String& storePath);
    virtual ~ContentRuleListStore();

    void compileContentRuleList(WTF::String&& identifier, WTF::String&& json, CompletionHandler<void(RefPtr<API::ContentRuleList>, std::error_code)>);
    void lookupContentRuleList(WTF::String&& identifier, CompletionHandler<void(RefPtr<API::ContentRuleList>, std::error_code)>);
    void removeContentRuleList(WTF::String&& identifier, CompletionHandler<void(std::error_code)>);

    void compileContentRuleListFile(WTF::String&& filePath, WTF::String&& identifier, WTF::String&& json, CompletionHandler<void(RefPtr<API::ContentRuleList>, std::error_code)>);
    void lookupContentRuleListFile(WTF::String&& filePath, WTF::String&& identifier, CompletionHandler<void(RefPtr<API::ContentRuleList>, std::error_code)>);
    void removeContentRuleListFile(WTF::String&& filePath, CompletionHandler<void(std::error_code)>);

    void getAvailableContentRuleListIdentifiers(CompletionHandler<void(WTF::Vector<WTF::String>)>);

    // For testing only.
    void synchronousRemoveAllContentRuleLists();
    void invalidateContentRuleListVersion(const WTF::String& identifier);
    void corruptContentRuleListHeader(const WTF::String& identifier, bool usingCurrentVersion);
    void corruptContentRuleListActionsMatchingEverything(const WTF::String& identifier);
    void invalidateContentRuleListHeader(const WTF::String& identifier);
    void getContentRuleListSource(WTF::String&& identifier, CompletionHandler<void(WTF::String)>);

private:
    WTF::String defaultStorePath();

    const WTF::String m_storePath;
    Ref<WTF::ConcurrentWorkQueue> m_compileQueue;
    Ref<WTF::WorkQueue> m_readQueue;
    Ref<WTF::WorkQueue> m_removeQueue;
#endif // ENABLE(CONTENT_EXTENSIONS)
};

const std::error_category& contentRuleListStoreErrorCategory();

inline std::error_code make_error_code(ContentRuleListStore::Error error)
{
    return { static_cast<int>(error), contentRuleListStoreErrorCategory() };
}

} // namespace API

namespace std {
template<> struct is_error_code_enum<API::ContentRuleListStore::Error> : public true_type { };
}
