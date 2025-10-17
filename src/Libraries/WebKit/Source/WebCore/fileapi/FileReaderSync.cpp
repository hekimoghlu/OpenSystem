/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#include "FileReaderSync.h"

#include "Blob.h"
#include "BlobURL.h"
#include "FileReaderLoader.h"
#include <JavaScriptCore/ArrayBuffer.h>

namespace WebCore {

FileReaderSync::FileReaderSync()
{
}

ExceptionOr<RefPtr<ArrayBuffer>> FileReaderSync::readAsArrayBuffer(ScriptExecutionContext& scriptExecutionContext, Blob& blob)
{
    FileReaderLoader loader(FileReaderLoader::ReadAsArrayBuffer, nullptr);
    auto result = startLoading(scriptExecutionContext, loader, blob);
    if (result.hasException())
        return result.releaseException();
    return loader.arrayBufferResult();
}

ExceptionOr<String> FileReaderSync::readAsBinaryString(ScriptExecutionContext& scriptExecutionContext, Blob& blob)
{
    FileReaderLoader loader(FileReaderLoader::ReadAsBinaryString, nullptr);
    return startLoadingString(scriptExecutionContext, loader, blob);
}

ExceptionOr<String> FileReaderSync::readAsText(ScriptExecutionContext& scriptExecutionContext, Blob& blob, const String& encoding)
{
    FileReaderLoader loader(FileReaderLoader::ReadAsText, nullptr);
    loader.setEncoding(encoding);
    return startLoadingString(scriptExecutionContext, loader, blob);
}

ExceptionOr<String> FileReaderSync::readAsDataURL(ScriptExecutionContext& scriptExecutionContext, Blob& blob)
{
    FileReaderLoader loader(FileReaderLoader::ReadAsDataURL, nullptr);
    loader.setDataType(blob.type());
    return startLoadingString(scriptExecutionContext, loader, blob);
}

ExceptionOr<void> FileReaderSync::startLoading(ScriptExecutionContext& scriptExecutionContext, FileReaderLoader& loader, Blob& blob)
{
    loader.start(&scriptExecutionContext, blob);
    auto error = loader.errorCode();
    if (!error)
        return { };
    
    return Exception { error.value() };
}

ExceptionOr<String> FileReaderSync::startLoadingString(ScriptExecutionContext& scriptExecutionContext, FileReaderLoader& loader, Blob& blob)
{
    auto result = startLoading(scriptExecutionContext, loader, blob);
    if (result.hasException())
        return result.releaseException();
    return loader.stringResult();
}

} // namespace WebCore
