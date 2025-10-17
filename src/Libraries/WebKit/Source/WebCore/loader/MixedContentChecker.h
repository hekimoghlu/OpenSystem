/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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

#include "FetchOptions.h"
#include "ResourceLoaderOptions.h"
#include <wtf/Forward.h>

namespace WebCore {

class LocalFrame;
class SecurityOrigin;
enum class Initiator : uint8_t;

namespace MixedContentChecker {

enum class ContentType {
    Active,
    ActiveCanWarn,
};

enum class ShouldLogWarning { No, Yes };

enum class IsUpgradable : bool { No, Yes, };

bool frameAndAncestorsCanRunInsecureContent(LocalFrame&, SecurityOrigin&, const URL&, ShouldLogWarning = ShouldLogWarning::Yes);
bool shouldUpgradeInsecureContent(LocalFrame&, IsUpgradable, const URL&, FetchOptions::Destination, Initiator);
bool shouldBlockRequestForDisplayableContent(LocalFrame&, const URL&, ContentType, IsUpgradable = IsUpgradable::No);
bool shouldBlockRequestForRunnableContent(LocalFrame&, SecurityOrigin&, const URL&, ShouldLogWarning = ShouldLogWarning::Yes);
void checkFormForMixedContent(LocalFrame&, const URL&);

WEBCORE_EXPORT bool canModifyRequest(const URL&, FetchOptions::Destination, Initiator);

} // namespace MixedContentChecker
} // namespace WebCore
