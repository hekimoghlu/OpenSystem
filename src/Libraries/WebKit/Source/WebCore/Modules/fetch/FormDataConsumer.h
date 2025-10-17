/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#include "ScriptExecutionContextIdentifier.h"
#include <span>
#include <wtf/Function.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class BlobLoader;
class FormData;
class ScriptExecutionContext;

class FormDataConsumer : public RefCountedAndCanMakeWeakPtr<FormDataConsumer> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(FormDataConsumer, WEBCORE_EXPORT);
public:
    using Callback = Function<bool(ExceptionOr<std::span<const uint8_t>>)>;
    static Ref<FormDataConsumer> create(const FormData& formData, ScriptExecutionContext& context, Callback&& callback) { return adoptRef(*new FormDataConsumer(formData, context, WTFMove(callback))); }
    WEBCORE_EXPORT ~FormDataConsumer();

    void start() { read(); }
    void cancel();

    bool hasPendingActivity() const { return !!m_blobLoader || m_isReadingFile; }

private:
    FormDataConsumer(const FormData&, ScriptExecutionContext&, Callback&&);

    void consumeData(const Vector<uint8_t>&);
    void consumeFile(const String&);
    void consumeBlob(const URL&);

    void consume(std::span<const uint8_t>);
    void read();
    void didFail(Exception&&);
    bool isCancelled() { return !m_context; }

    Ref<FormData> m_formData;
    RefPtr<ScriptExecutionContext> m_context;
    Callback m_callback;

    size_t m_currentElementIndex { 0 };
    Ref<WorkQueue> m_fileQueue;
    std::unique_ptr<BlobLoader> m_blobLoader;
    bool m_isReadingFile { false };
};

} // namespace WebCore
