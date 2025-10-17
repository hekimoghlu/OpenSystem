/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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

#include <wtf/Forward.h>

namespace WebCore {

class Page;
class PageGroup;
class SecurityOrigin;
class Storage;

namespace StorageEventDispatcher {
WEBCORE_EXPORT void dispatchSessionStorageEvents(const String& key, const String& oldValue, const String& newValue, Page&, const SecurityOrigin&, const String& url, const Function<bool(Storage&)>& isSourceStorage);
WEBCORE_EXPORT void dispatchLocalStorageEvents(const String& key, const String& oldValue, const String& newValue, PageGroup*, const SecurityOrigin&, const String& url, const Function<bool(Storage&)>& isSourceStorage);
}

} // namespace WebCore
