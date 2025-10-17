/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

#include "FrameIdentifier.h"
#include "PageIdentifier.h"
#include "ProcessQualified.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerClientType.h"
#include "ServiceWorkerTypes.h"
#include <wtf/URL.h>

namespace WebCore {

class SWClientConnection;
class ScriptExecutionContext;

enum class AdvancedPrivacyProtections : uint16_t;
enum class LastNavigationWasAppInitiated : bool { No, Yes };

struct ServiceWorkerClientData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    ScriptExecutionContextIdentifier identifier;
    ServiceWorkerClientType type;
    ServiceWorkerClientFrameType frameType;
    URL url;
    URL ownerURL;
    std::optional<PageIdentifier> pageIdentifier;
    std::optional<FrameIdentifier> frameIdentifier;
    LastNavigationWasAppInitiated lastNavigationWasAppInitiated;
    OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections;
    bool isVisible { false };
    bool isFocused { false };
    uint64_t focusOrder { 0 };
    Vector<String> ancestorOrigins;

    WEBCORE_EXPORT ServiceWorkerClientData isolatedCopy() const &;
    WEBCORE_EXPORT ServiceWorkerClientData isolatedCopy() &&;

    WEBCORE_EXPORT static ServiceWorkerClientData from(ScriptExecutionContext&);
};

using ServiceWorkerClientsMatchAllCallback = CompletionHandler<void(Vector<ServiceWorkerClientData>&&)>;

} // namespace WebCore
