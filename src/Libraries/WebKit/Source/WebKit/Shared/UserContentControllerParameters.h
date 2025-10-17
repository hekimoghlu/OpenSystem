/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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

#include "ContentWorldData.h"
#include "UserContentControllerIdentifier.h"
#include "WebCompiledContentRuleListData.h"
#include "WebUserContentControllerDataTypes.h"

namespace WebKit {

struct UserContentControllerParameters {
    UserContentControllerIdentifier identifier;
    Vector<ContentWorldData> userContentWorlds;
    Vector<WebUserScriptData> userScripts;
    Vector<WebUserStyleSheetData> userStyleSheets;
    Vector<WebScriptMessageHandlerData> messageHandlers;
#if ENABLE(CONTENT_EXTENSIONS)
    Vector<std::pair<WebCompiledContentRuleListData, URL>> contentRuleLists;
#endif
};

} // namespace WebKit
