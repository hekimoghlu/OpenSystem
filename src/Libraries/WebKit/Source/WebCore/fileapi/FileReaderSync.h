/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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

#include "ExceptionOr.h"

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {

class Blob;
class FileReaderLoader;
class ScriptExecutionContext;

class FileReaderSync : public RefCounted<FileReaderSync> {
public:
    static Ref<FileReaderSync> create()
    {
        return adoptRef(*new FileReaderSync);
    }

    ExceptionOr<RefPtr<JSC::ArrayBuffer>> readAsArrayBuffer(ScriptExecutionContext&, Blob&);
    ExceptionOr<String> readAsBinaryString(ScriptExecutionContext&, Blob&);
    ExceptionOr<String> readAsText(ScriptExecutionContext&, Blob&, const String& encoding);
    ExceptionOr<String> readAsDataURL(ScriptExecutionContext&, Blob&);

private:
    FileReaderSync();

    ExceptionOr<void> startLoading(ScriptExecutionContext&, FileReaderLoader&, Blob&);
    ExceptionOr<String> startLoadingString(ScriptExecutionContext&, FileReaderLoader&, Blob&);
};

} // namespace WebCore
