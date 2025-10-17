/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
#include "HTMLInputElementEntriesAPI.h"

#include "DOMFileSystem.h"
#include "ElementInlines.h"
#include "FileList.h"
#include "HTMLInputElement.h"

namespace WebCore {

using namespace HTMLNames;

Vector<Ref<FileSystemEntry>> HTMLInputElementEntriesAPI::webkitEntries(ScriptExecutionContext& context, HTMLInputElement& input)
{
    // As of Sept 2017, Chrome and Firefox both only populate webkitEntries when the webkitdirectory flag is unset.
    // We do the same for consistency.
    if (input.hasAttributeWithoutSynchronization(webkitdirectoryAttr))
        return { };

    RefPtr fileList = input.files();
    if (!fileList)
        return { };

    return fileList->files().map([&](auto& file) {
        return DOMFileSystem::createEntryForFile(context, file.copyRef());
    });
}

}
