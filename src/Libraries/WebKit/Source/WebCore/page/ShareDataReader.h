/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#include "ShareData.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {

class Blob;
class BlobLoader;
class Document;
class ScriptExecutionContext;

class ShareDataReader : public RefCounted<ShareDataReader> {
public:
    static Ref<ShareDataReader> create(CompletionHandler<void(ExceptionOr<ShareDataWithParsedURL&>)>&& completionHandler) { return adoptRef(*new ShareDataReader(WTFMove(completionHandler))); }
    ~ShareDataReader();
    void start(Document*, ShareDataWithParsedURL&&);
    void cancel();

private:
    explicit ShareDataReader(CompletionHandler<void(ExceptionOr<ShareDataWithParsedURL&>)>&&);
    void didFinishLoading(int, const String& fileName);

    CompletionHandler<void(ExceptionOr<ShareDataWithParsedURL&>)> m_completionHandler;
    ShareDataWithParsedURL m_shareData;
    int m_filesReadSoFar;
    Vector<UniqueRef<BlobLoader>> m_pendingFileLoads;
};

}
