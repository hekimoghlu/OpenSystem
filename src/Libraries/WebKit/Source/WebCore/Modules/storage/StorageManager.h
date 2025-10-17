/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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

#include "IDLTypes.h"
#include "StorageEstimate.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class FileSystemDirectoryHandle;
class NavigatorBase;
template<typename> class DOMPromiseDeferred;
template<typename> class ExceptionOr;

class StorageManager : public RefCounted<StorageManager> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StorageManager);
public:
    static Ref<StorageManager> create(NavigatorBase&);
    ~StorageManager();
    void persisted(DOMPromiseDeferred<IDLBoolean>&&);
    void persist(DOMPromiseDeferred<IDLBoolean>&&);
    using Estimate = StorageEstimate;
    void estimate(DOMPromiseDeferred<IDLDictionary<Estimate>>&&);
    void fileSystemAccessGetDirectory(DOMPromiseDeferred<IDLInterface<FileSystemDirectoryHandle>>&&);

private:
    explicit StorageManager(NavigatorBase&);
    WeakPtr<NavigatorBase> m_navigator;
};

} // namespace WebCore
