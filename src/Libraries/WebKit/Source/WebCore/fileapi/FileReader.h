/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#include "ActiveDOMObject.h"
#include "DOMException.h"
#include "EventTarget.h"
#include "ExceptionCode.h"
#include "ExceptionOr.h"
#include "FileReaderLoader.h"
#include "FileReaderLoaderClient.h"
#include "FileReaderSync.h"
#include <wtf/HashMap.h>
#include <wtf/UniqueRef.h>

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {

class Blob;

class FileReader final : public RefCounted<FileReader>, public ActiveDOMObject, public EventTarget, private FileReaderLoaderClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FileReader);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<FileReader> create(ScriptExecutionContext&);

    virtual ~FileReader();

    enum ReadyState {
        EMPTY = 0,
        LOADING = 1,
        DONE = 2
    };

    ExceptionOr<void> readAsArrayBuffer(Blob&);
    ExceptionOr<void> readAsBinaryString(Blob&);
    ExceptionOr<void> readAsText(Blob&, const String& encoding);
    ExceptionOr<void> readAsDataURL(Blob&);
    void abort();

    void doAbort();

    ReadyState readyState() const { return m_state; }
    DOMException* error() { return m_error.get(); }
    FileReaderLoader::ReadType readType() const { return m_readType; }
    std::optional<std::variant<String, RefPtr<JSC::ArrayBuffer>>> result() const;

private:
    explicit FileReader(ScriptExecutionContext&);

    // ActiveDOMObject.
    void stop() final;
    bool virtualHasPendingActivity() const final;

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::FileReader; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    void enqueueTask(Function<void()>&&);

    void didStartLoading() final;
    void didReceiveData() final;
    void didFinishLoading() final;
    void didFail(ExceptionCode errorCode) final;

    ExceptionOr<void> readInternal(Blob&, FileReaderLoader::ReadType);
    void fireErrorEvent(int httpStatusCode);
    void fireEvent(const AtomString& type);

    ReadyState m_state { EMPTY };
    bool m_finishedLoading { false };
    RefPtr<Blob> m_blob;
    FileReaderLoader::ReadType m_readType { FileReaderLoader::ReadAsBinaryString };
    String m_encoding;
    std::unique_ptr<FileReaderLoader> m_loader;
    RefPtr<DOMException> m_error;
    MonotonicTime m_lastProgressNotificationTime { MonotonicTime::nan() };
    HashMap<uint64_t, Function<void()>> m_pendingTasks;
};

} // namespace WebCore
